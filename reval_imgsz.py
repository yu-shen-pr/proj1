import argparse
import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional


def _infer_channels_from_loaded_model(yolo: Any) -> Optional[int]:
    try:
        import torch

        m = getattr(yolo, "model", None)
        if m is None:
            return None

        for mod in m.modules():
            if isinstance(mod, torch.nn.Conv2d):
                return int(mod.in_channels)
        return None
    except Exception:
        return None


def _patch_validator_warmup_channels_dynamic() -> None:
    try:
        from ultralytics.engine.validator import BaseValidator  # type: ignore
    except Exception:
        return

    def _infer_warmup_channels_from_model(model: Any) -> Optional[int]:
        try:
            import torch

            # AutoBackend wraps the real model under `.model`
            root = getattr(model, "model", None)
            if root is None:
                root = model

            # Ultralytics DetectionModel keeps layers in `root.model` (list/ModuleList)
            mm = getattr(root, "model", None)
            try:
                if mm is not None and hasattr(mm, "__getitem__"):
                    m0 = mm[0]
                    conv = getattr(m0, "conv", None)
                    if isinstance(conv, torch.nn.Conv2d):
                        return int(conv.in_channels)
            except Exception:
                pass

            ins: list[int] = []
            best = None
            for mod in root.modules():
                if not isinstance(mod, torch.nn.Conv2d):
                    continue
                ins.append(int(mod.in_channels))
                k = getattr(mod, "kernel_size", None)
                ks = tuple(int(x) for x in k) if isinstance(k, tuple) else None
                cand = (int(mod.out_channels), int(mod.in_channels), ks)
                if cand[0] == 16 and cand[2] == (3, 3):
                    best = cand
            if ins:
                mn = int(min(ins))
                if mn in (3, 6):
                    return mn
            if best is not None:
                return int(best[1])
            return None
        except Exception:
            return None

    if getattr(BaseValidator, "_reval_dyn_ch_patched", False):
        return

    orig_call = BaseValidator.__call__

    def __call__(self, *args, **kwargs):  # type: ignore
        model = kwargs.get("model", None)
        if model is None and len(args) >= 1:
            model = args[0]
        try:
            ch = _infer_warmup_channels_from_model(model) if model is not None else None
            if ch is not None and hasattr(self, "data") and isinstance(self.data, dict):
                self.data["channels"] = int(ch)
        except Exception:
            pass
        return orig_call(self, *args, **kwargs)

    BaseValidator.__call__ = __call__  # type: ignore
    BaseValidator._reval_dyn_ch_patched = True  # type: ignore


def _temporary_patch_yolodataset_load_image_for_6ch(motion_dirname: str = "motion_images") -> None:
    try:
        from ultralytics.data.dataset import YOLODataset  # type: ignore
    except Exception:
        return

    if not hasattr(YOLODataset, "_reval_orig_load_image"):
        YOLODataset._reval_orig_load_image = YOLODataset.load_image  # type: ignore

    orig_load_image = YOLODataset._reval_orig_load_image  # type: ignore

    def _motion_path_from_image_path(img_path: str) -> str:
        p = img_path.replace("\\", os.sep).replace("/", os.sep)
        token = os.sep + "images" + os.sep
        if token in p:
            p = p.replace(token, os.sep + motion_dirname + os.sep, 1)
        else:
            parts = p.split(os.sep)
            for i in range(len(parts) - 1):
                if parts[i] == "images":
                    parts[i] = motion_dirname
                    p = os.sep.join(parts)
                    break
        return p

    def load_image_6ch(self, i: int, *args, **kwargs):  # type: ignore
        out = orig_load_image(self, i, *args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 1:
            im = out[0]
        else:
            im = out

        img_path = self.im_files[i]
        motion_path = _motion_path_from_image_path(img_path)

        import cv2
        import numpy as np

        motion = cv2.imread(motion_path, cv2.IMREAD_UNCHANGED)
        if motion is None:
            motion = np.zeros((im.shape[0], im.shape[1], 3), dtype=im.dtype)
        else:
            if motion.ndim == 2:
                motion = np.stack([motion, motion, motion], axis=-1)
            if motion.shape[0] != im.shape[0] or motion.shape[1] != im.shape[1]:
                motion = cv2.resize(motion, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            if motion.shape[2] > 3:
                motion = motion[:, :, :3]
            if motion.shape[2] == 1:
                motion = np.repeat(motion, 3, axis=2)

        im6 = np.concatenate([im, motion], axis=2)

        if isinstance(out, tuple) and len(out) >= 1:
            out = (im6,) + out[1:]
        else:
            out = im6
        return out

    YOLODataset.load_image = load_image_6ch  # type: ignore


def _restore_yolodataset_load_image() -> None:
    try:
        from ultralytics.data.dataset import YOLODataset  # type: ignore
    except Exception:
        return

    orig = getattr(YOLODataset, "_reval_orig_load_image", None)
    if orig is None:
        return
    YOLODataset.load_image = orig  # type: ignore


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
    _temporary_patch_yolodataset_load_image_for_6ch(motion_dirname=motion_dirname)


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
    model_in_channels: Optional[int]
    imgsz: str
    rect: bool
    batch: int
    device: str
    P: Optional[float]
    R: Optional[float]
    mAP50: Optional[float]
    mAP50_95: Optional[float]
    inf_ms: Optional[float]
    error: str


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

    rect = False
    val_imgsz: Any = imgsz_obj
    if isinstance(imgsz_obj, list) and len(imgsz_obj) == 2:
        rect = True
        try:
            val_imgsz = int(max(int(imgsz_obj[0]), int(imgsz_obj[1])))
        except Exception:
            val_imgsz = 640

    from ultralytics import YOLO

    _patch_validator_warmup_channels_dynamic()

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

        yolo = YOLO(best_pt)
        ch_pt = _infer_channels_from_pt(best_pt)
        ch_model = _infer_channels_from_loaded_model(yolo)
        ch = ch_model if ch_model is not None else ch_pt
        if ch == 6:
            _ensure_6ch_patches(motion_dirname=str(args.motion_dirname))
        else:
            _restore_yolodataset_load_image()

        err = ""
        out = None
        try:
            out = yolo.val(
                data=data,
                imgsz=val_imgsz,
                rect=bool(rect),
                batch=int(args.batch),
                device=str(args.device),
                project="reval",
                name=os.path.basename(rd.rstrip(os.sep)),
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        finally:
            if ch == 6:
                _restore_yolodataset_load_image()

        m = _extract_metrics(out) if out is not None else {"P": None, "R": None, "mAP50": None, "mAP50-95": None}
        inf_ms = _extract_speed_ms(out) if out is not None else None

        rows.append(
            ReValRow(
                run_dir=rd,
                run_name=os.path.basename(rd.rstrip(os.sep)),
                best_pt=best_pt,
                data=data,
                channels=ch,
                model_in_channels=ch_model,
                imgsz=str(args.imgsz),
                rect=bool(rect),
                batch=int(args.batch),
                device=str(args.device),
                P=m.get("P"),
                R=m.get("R"),
                mAP50=m.get("mAP50"),
                mAP50_95=m.get("mAP50-95"),
                inf_ms=inf_ms,
                error=err,
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
