import argparse
import csv
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _try_import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _read_reval_metrics(reval_csv: str) -> dict[str, dict[str, Any]]:
    """Read unified re-validation CSV produced by reval_imgsz.py.

    Returns mapping: run_name -> {P, R, mAP50, mAP50_95, imgsz, rect, batch, device, inf_ms, error}
    """

    if not reval_csv or not os.path.isfile(reval_csv):
        return {}

    rows = _read_csv_rows(reval_csv)
    if not rows:
        return {}

    def pick(d: dict[str, str], *keys: str) -> str:
        for k in keys:
            if k in d and str(d.get(k, "")).strip() != "":
                return str(d.get(k, ""))
        return ""

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        run_name = pick(r, "run_name", "name", "Run")
        if not run_name:
            continue

        out[run_name] = {
            "P": _safe_float(pick(r, "P", "p")),
            "R": _safe_float(pick(r, "R", "r")),
            "mAP50": _safe_float(pick(r, "mAP50", "map50")),
            "mAP50_95": _safe_float(pick(r, "mAP50_95", "mAP50-95", "map50_95", "map50-95")),
            "imgsz": pick(r, "imgsz"),
            "rect": pick(r, "rect"),
            "batch": pick(r, "batch"),
            "device": pick(r, "device"),
            "inf_ms": _safe_float(pick(r, "inf_ms", "inference_ms", "speed")),
            "error": pick(r, "error"),
        }
    return out


def _read_yaml(path: str) -> dict[str, Any]:
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


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _detect_metric_columns(header: list[str]) -> dict[str, str]:
    h = [c.strip() for c in header]
    def find_one(cands: list[str]) -> str:
        for cand in cands:
            for col in h:
                if col == cand:
                    return col
        for cand in cands:
            for col in h:
                if cand in col:
                    return col
        return ""

    return {
        "P": find_one(["metrics/precision(B)", "precision", "metrics/precision"]),
        "R": find_one(["metrics/recall(B)", "recall", "metrics/recall"]),
        "mAP50": find_one(["metrics/mAP50(B)", "mAP50", "metrics/mAP50"]),
        "mAP50-95": find_one(["metrics/mAP50-95(B)", "mAP50-95", "metrics/mAP50-95"]),
    }


def _read_best_metrics_from_results_csv(path: str) -> dict[str, Optional[float]]:
    rows = _read_csv_rows(path)
    if not rows:
        return {"P": None, "R": None, "mAP50": None, "mAP50-95": None}

    cols = _detect_metric_columns(list(rows[0].keys()))

    def score(row: dict[str, str]) -> float:
        key = cols.get("mAP50-95", "")
        return _safe_float(row.get(key)) or float("-inf")

    best = max(rows, key=score)

    out: dict[str, Optional[float]] = {}
    for k in ["P", "R", "mAP50", "mAP50-95"]:
        col = cols.get(k, "")
        out[k] = _safe_float(best.get(col)) if col else None
    return out


def _infer_channels_from_best_pt(best_pt: str) -> Optional[int]:
    try:
        import torch

        ckpt = torch.load(best_pt, map_location="cpu")
        state = None
        if isinstance(ckpt, dict):
            if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                state = ckpt["model"].state_dict()
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "ema" in ckpt and hasattr(ckpt["ema"], "state_dict"):
                state = ckpt["ema"].state_dict()
        if not isinstance(state, dict):
            return None

        for k, v in state.items():
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            if len(shape) == 4 and shape[0] == 16 and shape[2] == 3 and shape[3] == 3:
                return int(shape[1])
        for k, v in state.items():
            if not hasattr(v, "shape"):
                continue
            shape = tuple(int(x) for x in v.shape)
            if len(shape) == 4 and shape[2] == 3 and shape[3] == 3:
                return int(shape[1])
        return None
    except Exception:
        return None


@dataclass
class RunSummary:
    run_dir: str
    run_name: str
    project: str
    data: str
    model: str
    imgsz: Any
    epochs: Any
    batch: Any
    device: Any
    channels: Optional[int]
    P: Optional[float]
    R: Optional[float]
    mAP50: Optional[float]
    mAP50_95: Optional[float]


def _format_float(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return ""
    return f"{x:.{nd}f}"


def _find_runs(runs_dir: str) -> list[str]:
    out: list[str] = []
    for root, dirs, files in os.walk(runs_dir):
        if "results.csv" in files:
            out.append(root)
    return sorted(set(out))


def _summarize_one(run_dir: str) -> Optional[RunSummary]:
    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(results_csv):
        return None

    args_yaml = os.path.join(run_dir, "args.yaml")
    args = _read_yaml(args_yaml) if os.path.isfile(args_yaml) else {}

    weights_best = os.path.join(run_dir, "weights", "best.pt")
    ch = _infer_channels_from_best_pt(weights_best) if os.path.isfile(weights_best) else None

    m = _read_best_metrics_from_results_csv(results_csv)

    run_name = os.path.basename(run_dir.rstrip(os.sep))
    project = os.path.basename(os.path.dirname(run_dir.rstrip(os.sep)))

    return RunSummary(
        run_dir=run_dir,
        run_name=run_name,
        project=str(args.get("project", project)),
        data=str(args.get("data", "")),
        model=str(args.get("model", "")),
        imgsz=args.get("imgsz", ""),
        epochs=args.get("epochs", ""),
        batch=args.get("batch", ""),
        device=args.get("device", ""),
        channels=ch,
        P=m.get("P"),
        R=m.get("R"),
        mAP50=m.get("mAP50"),
        mAP50_95=m.get("mAP50-95"),
    )


def _write_csv(path: str, rows: list[RunSummary]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _write_markdown_table(path: str, rows: list[RunSummary]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cols = [
        ("Run", lambda r: r.run_name),
        ("Ch", lambda r: "" if r.channels is None else str(r.channels)),
        ("P", lambda r: _format_float(r.P)),
        ("R", lambda r: _format_float(r.R)),
        ("mAP50", lambda r: _format_float(r.mAP50)),
        ("mAP50-95", lambda r: _format_float(r.mAP50_95)),
        ("imgsz", lambda r: str(r.imgsz)),
        ("epochs", lambda r: str(r.epochs)),
        ("batch", lambda r: str(r.batch)),
        ("data", lambda r: str(r.data)),
    ]

    header = "| " + " | ".join([c[0] for c in cols]) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"

    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([c[1](r).replace("|", "\\|") for c in cols]) + " |\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.writelines(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--out_dir", type=str, default="result_summary")
    ap.add_argument("--filter", type=str, default="")
    ap.add_argument("--reval_csv", type=str, default="")
    ap.add_argument("--display_imgsz", type=str, default="")
    args = ap.parse_args()

    runs_dir = os.path.abspath(args.runs_dir)
    run_dirs = _find_runs(runs_dir)

    reval_map = _read_reval_metrics(str(args.reval_csv)) if args.reval_csv else {}

    rows: list[RunSummary] = []
    for rd in run_dirs:
        if args.filter and args.filter not in rd:
            continue
        s = _summarize_one(rd)
        if s is not None:
            # Prefer unified re-validation metrics if provided
            if s.run_name in reval_map:
                rv = reval_map[s.run_name]
                s.P = rv.get("P") if rv.get("P") is not None else s.P
                s.R = rv.get("R") if rv.get("R") is not None else s.R
                s.mAP50 = rv.get("mAP50") if rv.get("mAP50") is not None else s.mAP50
                s.mAP50_95 = rv.get("mAP50_95") if rv.get("mAP50_95") is not None else s.mAP50_95
                if rv.get("imgsz"):
                    s.imgsz = rv.get("imgsz")

            if args.display_imgsz:
                s.imgsz = str(args.display_imgsz)
            rows.append(s)

    rows.sort(key=lambda r: (r.project, r.run_name))

    if not rows:
        print(f"No runs found under: {runs_dir}")
        return 1

    out_csv = os.path.join(args.out_dir, "all_runs_summary.csv")
    out_md = os.path.join(args.out_dir, "main_table.md")
    _write_csv(out_csv, rows)
    _write_markdown_table(out_md, rows)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
