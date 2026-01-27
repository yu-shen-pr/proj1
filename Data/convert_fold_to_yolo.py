import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import yaml

# NOTE: User modification on top of the original repository codebase:
# added a fold-to-YOLO conversion script and made the conversion robust to
# sequences missing IR_label.json (skips and reports them instead of crashing).


def _read_list_file(list_path: Path) -> list[Path]:
    lines = list_path.read_text(encoding="utf-8").splitlines()
    out: list[Path] = []
    for line in lines:
        p = line.strip()
        if not p:
            continue
        out.append(Path(p))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return

    _ensure_dir(dst.parent)

    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            # fallback to copy
            pass

    shutil.copy2(src, dst)


def _yolo_line_from_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> str:
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    ww = w / float(img_w)
    hh = h / float(img_h)

    # clip to [0, 1]
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    ww = max(0.0, min(1.0, ww))
    hh = max(0.0, min(1.0, hh))

    return f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}"


def _load_ir_label(seq_dir: Path) -> dict:
    p = seq_dir / "IR_label.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _frame_index_from_name(img_path: Path) -> int:
    # 000001.jpg -> index 0
    stem = img_path.stem
    try:
        return int(stem) - 1
    except ValueError as e:
        raise ValueError(f"Unexpected frame name: {img_path.name}") from e


def convert_split(
    *,
    project_root: Path,
    split_name: str,
    img_rel_paths: list[Path],
    out_root: Path,
    link_mode: str,
) -> tuple[int, int]:
    out_images = out_root / "images" / split_name
    out_labels = out_root / "labels" / split_name
    _ensure_dir(out_images)
    _ensure_dir(out_labels)

    # cache seq labels to avoid re-reading JSON for every frame
    seq_cache: dict[str, dict] = {}
    seq_img_hw: dict[str, tuple[int, int]] = {}
    missing_seq_labels: set[str] = set()

    processed = 0
    missing = 0

    for rel in img_rel_paths:
        # list files contain ./Data/train/...; resolve relative to project root
        rel_str = str(rel).replace("\\", "/")
        if rel_str.startswith("./"):
            rel_str = rel_str[2:]
        src_img = (project_root / rel_str).resolve()

        if not src_img.exists():
            missing += 1
            continue

        seq_dir = src_img.parent
        seq_name = seq_dir.name

        if seq_name not in seq_cache:
            seq_cache[seq_name] = _load_ir_label(seq_dir)

        if not seq_cache[seq_name]:
            missing_seq_labels.add(seq_name)
            missing += 1
            continue

        if seq_name not in seq_img_hw:
            im = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise RuntimeError(f"Failed to read image: {src_img}")
            h, w = im.shape[:2]
            seq_img_hw[seq_name] = (w, h)

        labels = seq_cache[seq_name]
        w, h = seq_img_hw[seq_name]
        idx = _frame_index_from_name(src_img)

        exist_list = labels.get("exist")
        rect_list = labels.get("gt_rect")

        if exist_list is None or rect_list is None:
            missing += 1
            continue

        if idx < 0 or idx >= len(exist_list) or idx >= len(rect_list):
            missing += 1
            continue

        out_img = out_images / f"{seq_name}__{src_img.name}"
        out_txt = out_labels / f"{seq_name}__{src_img.stem}.txt"

        _safe_link_or_copy(src_img, out_img, link_mode)

        exist = int(exist_list[idx])
        rect = rect_list[idx]

        # YOLO expects an empty txt file if no object exists
        if exist == 0:
            out_txt.write_text("", encoding="utf-8")
        else:
            if not isinstance(rect, list) or len(rect) != 4:
                out_txt.write_text("", encoding="utf-8")
            else:
                x, y, bw, bh = rect
                # Some datasets may have invalid rectangles; guard here
                if bw <= 0 or bh <= 0:
                    out_txt.write_text("", encoding="utf-8")
                else:
                    out_txt.write_text(_yolo_line_from_xywh(float(x), float(y), float(bw), float(bh), w, h) + "\n", encoding="utf-8")

        processed += 1

    if missing_seq_labels:
        print(f"[{split_name}] missing IR_label.json for sequences: {sorted(missing_seq_labels)}")

    return processed, missing


def write_data_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["uav"],
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--best_folds_dir", type=str, default="Data/best_folds")
    parser.add_argument("--out", type=str, default="Data/yolo_fold4")
    parser.add_argument("--mode", type=str, choices=["link", "copy"], default="link")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    best = project_root / args.best_folds_dir

    train_list = best / f"train_fold{args.fold}_newdata.txt"
    val_list = best / f"valid_fold{args.fold}_newdata.txt"

    if not train_list.exists():
        raise FileNotFoundError(f"Missing train list: {train_list}")
    if not val_list.exists():
        raise FileNotFoundError(f"Missing val list: {val_list}")

    out_root = (project_root / args.out).resolve()
    _ensure_dir(out_root)

    train_imgs = _read_list_file(train_list)
    val_imgs = _read_list_file(val_list)

    train_processed, train_missing = convert_split(
        project_root=project_root,
        split_name="train",
        img_rel_paths=train_imgs,
        out_root=out_root,
        link_mode=args.mode,
    )
    val_processed, val_missing = convert_split(
        project_root=project_root,
        split_name="val",
        img_rel_paths=val_imgs,
        out_root=out_root,
        link_mode=args.mode,
    )

    write_data_yaml(out_root)

    print(f"YOLO dataset created at: {out_root}")
    print(f"train: processed={train_processed}, missing={train_missing}")
    print(f"val: processed={val_processed}, missing={val_missing}")
    print(f"data.yaml: {out_root / 'data.yaml'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
