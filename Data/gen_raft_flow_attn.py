import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor


def _parse_seq_and_index(name: str) -> tuple[str, int] | None:
    stem = Path(name).stem
    if "__" not in stem:
        return None
    seq, frame = stem.split("__", 1)
    try:
        idx = int(frame)
    except ValueError:
        return None
    return seq, idx


def _tensor_gray_u8(img_rgb: Image.Image) -> torch.Tensor:
    img_gray = img_rgb.convert("L")
    return torch.from_numpy(np.array(img_gray, dtype=np.uint8))


def _quantile_or_max(x: torch.Tensor, q: float) -> float:
    x = x.flatten()
    if x.numel() == 0:
        return 0.0
    try:
        return float(torch.quantile(x, q).item())
    except Exception:
        return float(x.max().item())


def _flow_to_uint8(flow: torch.Tensor, q: float = 0.99) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # flow: [2, H, W] on CPU
    u = flow[0]
    v = flow[1]

    max_abs = max(_quantile_or_max(u.abs(), q), _quantile_or_max(v.abs(), q), 1e-6)

    u01 = (u / max_abs).clamp(-1.0, 1.0) * 0.5 + 0.5
    v01 = (v / max_abs).clamp(-1.0, 1.0) * 0.5 + 0.5

    u8 = (u01 * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    v8 = (v01 * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

    mag = torch.sqrt(u * u + v * v)
    mag_scale = max(_quantile_or_max(mag, q), 1e-6)
    attn = (mag / mag_scale).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

    return u8, v8, attn


def _apply_attn(gray_u8: np.ndarray, u8: np.ndarray, v8: np.ndarray, attn: np.ndarray, alpha: float) -> np.ndarray:
    g = gray_u8.astype(np.float32) / 255.0
    u = u8.astype(np.float32) / 255.0
    v = v8.astype(np.float32) / 255.0

    a = np.clip(attn, 0.0, 1.0)
    gain = 1.0 + alpha * a

    g = np.clip(g * gain, 0.0, 1.0)
    u = np.clip(u * a, 0.0, 1.0)
    v = np.clip(v * a, 0.0, 1.0)

    out = np.stack(
        [
            (g * 255.0).round().astype(np.uint8),
            (v * 255.0).round().astype(np.uint8),
            (u * 255.0).round().astype(np.uint8),
        ],
        axis=-1,
    )
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(out_root: Path, class_names: list[str]) -> None:
    yaml_path = out_root / "data.yaml"
    names_lines = "\n".join([f"- {n}" for n in class_names])
    yaml_path.write_text(
        f"path: {out_root.as_posix()}\ntrain: images/train\nval: images/val\nnc: {len(class_names)}\nnames:\n{names_lines}\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="Data/yolo_fold4")
    ap.add_argument("--out_root", type=str, default="Data/yolo_fold4_raft_attn")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--flow_q", type=float, default=0.99)
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    weights = Raft_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = raft_large(weights=weights, progress=True).to(device).eval()

    for split in ["train", "val"]:
        in_images = in_root / "images" / split
        in_labels = in_root / "labels" / split
        out_images = out_root / "images" / split
        out_labels = out_root / "labels" / split
        _ensure_dir(out_images)
        _ensure_dir(out_labels)

        img_paths = sorted([p for p in in_images.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])

        seq_to_frames: dict[str, list[tuple[int, Path]]] = {}
        for p in img_paths:
            parsed = _parse_seq_and_index(p.name)
            if parsed is None:
                continue
            seq, idx = parsed
            seq_to_frames.setdefault(seq, []).append((idx, p))

        for seq in seq_to_frames:
            seq_to_frames[seq].sort(key=lambda x: x[0])

        with torch.inference_mode():
            for seq, frames in tqdm(seq_to_frames.items(), desc=f"RAFT {split}"):
                prev_img_p: Path | None = None

                for _, cur_img_p in frames:
                    out_img_p = out_images / cur_img_p.name
                    out_lbl_p = out_labels / (cur_img_p.stem + ".txt")

                    src_lbl = in_labels / (cur_img_p.stem + ".txt")
                    if src_lbl.exists():
                        out_lbl_p.write_bytes(src_lbl.read_bytes())
                    else:
                        out_lbl_p.write_text("", encoding="utf-8")

                    cur_img = Image.open(cur_img_p).convert("RGB")
                    gray_u8 = np.array(cur_img.convert("L"), dtype=np.uint8)

                    if prev_img_p is None:
                        u8 = np.full_like(gray_u8, 128, dtype=np.uint8)
                        v8 = np.full_like(gray_u8, 128, dtype=np.uint8)
                        attn = np.zeros_like(gray_u8, dtype=np.float32)
                        merged = _apply_attn(gray_u8, u8, v8, attn, float(args.alpha))
                        Image.fromarray(merged).save(out_img_p)
                        prev_img_p = cur_img_p
                        continue

                    prev_img = Image.open(prev_img_p).convert("RGB")

                    t1 = to_tensor(prev_img)
                    t2 = to_tensor(cur_img)
                    t1, t2 = preprocess(t1, t2)

                    h0, w0 = cur_img.size[1], cur_img.size[0]
                    h1, w1 = t2.shape[-2], t2.shape[-1]

                    pred = model(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
                    flow = pred[-1][0].detach().float().cpu()  # [2, h1, w1]

                    if (h1, w1) != (h0, w0):
                        flow = flow.unsqueeze(0)
                        flow = F.interpolate(flow, size=(h0, w0), mode="bilinear", align_corners=False)
                        scale_x = w0 / float(w1)
                        scale_y = h0 / float(h1)
                        flow[:, 0] *= scale_x
                        flow[:, 1] *= scale_y
                        flow = flow[0]

                    u8, v8, attn = _flow_to_uint8(flow, q=float(args.flow_q))
                    merged = _apply_attn(gray_u8, u8, v8, attn, float(args.alpha))
                    Image.fromarray(merged).save(out_img_p)

                    prev_img_p = cur_img_p

    _write_dataset_yaml(out_root, ["uav"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
