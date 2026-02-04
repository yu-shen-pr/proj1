import argparse
import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional


def _parse_imgsz(s: str) -> Any:
    s = str(s).strip()
    if not s:
        return 640
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," in s:
        a, b = s.split(",", 1)
        return [int(a.strip()), int(b.strip())]
    return int(float(s))


def _try_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _read_yaml(path: str) -> dict:
    y = _try_import_yaml()
    if y is None:
        out: dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip("'\"")
        return out

    with open(path, "r", encoding="utf-8") as f:
        d = y.safe_load(f)
    return d if isinstance(d, dict) else {}


def _infer_channels_from_pt(best_pt: str) -> Optional[int]:
    try:
        import torch

        ckpt = torch.load(best_pt, map_location="cpu")
        state = None
        if isinstance(ckpt, dict):
            if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                state = ckpt["model"].state_dict()
            elif "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
                state = ckpt["ema"].state_dict()
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
        if not isinstance(state, dict):
            return None

        cand = []
        for k, v in state.items():
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            if len(shape) == 4 and shape[0] == 16 and shape[2] == 3 and shape[3] == 3:
                cand.append((k, shape[1]))
        if cand:
            cand.sort(key=lambda x: x[0])
            return int(cand[0][1])

        for k, v in state.items():
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            if len(shape) == 4 and shape[2] == 3 and shape[3] == 3:
                return int(shape[1])
        return None
    except Exception:
        return None


def _ensure_6ch_patches(motion_dirname: str = "motion_images") -> None:
    import train_yolo_6ch as t6

    t6._patch_yolodataset_for_motion(motion_dirname=motion_dirname)
    t6._patch_check_det_dataset_channels(in_ch=6)
    t6._patch_validator_warmup_channels(in_ch=6)


def _extract_metrics(val_out: Any) -> dict[str, Optional[float]]:
    def sf(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    rd = getattr(val_out, "results_dict", None)
    if isinstance(rd, dict):
        return {
            "P": sf(rd.get("metrics/precision(B)") or rd.get("precision") or rd.get("metrics/precision")),
            "R": sf(rd.get("metrics/recall(B)") or rd.get("recall") or rd.get("metrics/recall")),
            "mAP50": sf(rd.get("metrics/mAP50(B)") or rd.get("mAP50") or rd.get("metrics/mAP50")),
            "mAP50-95": sf(rd.get("metrics/mAP50-95(B)") or rd.get("mAP50-95") or rd.get("metrics/mAP50-95")),
        }

    box = getattr(val_out, "box", None)
    if box is not None:
        return {
            "P": sf(getattr(box, "mp", None)),
            "R": sf(getattr(box, "mr", None)),
            "mAP50": sf(getattr(box, "map50", None)),
            "mAP50-95": sf(getattr(box, "map", None)),
        }

    return {"P": None, "R": None, "mAP50": None, "mAP50-95": None}


def _extract_speed_ms(val_out: Any) -> Optional[float]:
    speed = getattr(val_out, "speed", None)
    if isinstance(speed, dict):
        try:
            inf = speed.get("inference", None)
            if inf is not None:
                return float(inf)
        except Exception:
            pass
    return None


@dataclass
class ReValRow:
    run_dir: str
    run_name: str
    best_pt: str
    data: str
    channels: Optional[int]
    imgsz: str
    batch: int
    device: str
    P: Optional[float]
    R: Optional[float]
    mAP50: Optional[float]
    mAP50_95: Optional[float]
    inf_ms: Optional[float]


def _find_run_dirs(runs_dir: str) -> list[str]:
    out: list[str] = []
    for root, dirs, files in os.walk(runs_dir):
        if "weights" in dirs and os.path.isfile(os.path.join(root, "weights", "best.pt")):
            out.append(root)
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--filter", type=str, default="")
    ap.add_argument("--imgsz", type=str, default="640,512")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--motion_dirname", type=str, default="motion_images")
    ap.add_argument("--out_csv", type=str, default="result_summary/reval_imgsz.csv")
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    imgsz_obj = _parse_imgsz(args.imgsz)

    from ultralytics import YOLO

    run_dirs = _find_run_dirs(runs_dir)
    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter in d]

    rows: list[ReValRow] = []

    for rd in run_dirs:
        best_pt = os.path.join(rd, "weights", "best.pt")
        args_yaml = os.path.join(rd, "args.yaml")
        cfg = _read_yaml(args_yaml) if os.path.isfile(args_yaml) else {}
        data = str(cfg.get("data", ""))
        if not data:
            continue

        ch = _infer_channels_from_pt(best_pt)
        if ch == 6:
            _ensure_6ch_patches(motion_dirname=str(args.motion_dirname))

        yolo = YOLO(best_pt)
        out = yolo.val(
            data=data,
            imgsz=imgsz_obj,
            batch=int(args.batch),
            device=str(args.device),
            project="reval",
            name=os.path.basename(rd.rstrip(os.sep)),
        )

        m = _extract_metrics(out)
        inf_ms = _extract_speed_ms(out)

        rows.append(
            ReValRow(
                run_dir=rd,
                run_name=os.path.basename(rd.rstrip(os.sep)),
                best_pt=best_pt,
                data=data,
                channels=ch,
                imgsz=str(args.imgsz),
                batch=int(args.batch),
                device=str(args.device),
                P=m.get("P"),
                R=m.get("R"),
                mAP50=m.get("mAP50"),
                mAP50_95=m.get("mAP50-95"),
                inf_ms=inf_ms,
            )
        )

    if not rows:
        print(f"No runnable runs found under: {runs_dir}")
        return 1

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    print(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
