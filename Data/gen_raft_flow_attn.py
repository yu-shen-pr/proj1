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


def _compute_attn_mag(flow: torch.Tensor, q: float, *, gamma: float = 1.0, t0: float = 0.0) -> torch.Tensor:
    u = flow[0]
    v = flow[1]
    mag = torch.sqrt(u * u + v * v)
    mag_scale = max(_quantile_or_max(mag, q), 1e-6)
    a = (mag / mag_scale).clamp(0.0, 1.0)
    t0 = float(t0)
    if t0 > 0.0:
        t0 = min(t0, 0.999)
        a = ((a - t0) / (1.0 - t0)).clamp(0.0, 1.0)
    gamma = float(gamma)
    if gamma != 1.0:
        a = a.clamp(0.0, 1.0).pow(gamma)
    return a


def _compute_attn_coherence_gate(
    flow: torch.Tensor,
    q: float,
    *,
    coh_tau: float,
    coh_k: float,
    mag_gamma: float = 1.0,
    mag_t0: float = 0.0,
) -> torch.Tensor:
    u = flow[0:1]
    v = flow[1:2]

    mag = torch.sqrt(u * u + v * v)

    u_mean = F.avg_pool2d(u, kernel_size=3, stride=1, padding=1)
    v_mean = F.avg_pool2d(v, kernel_size=3, stride=1, padding=1)

    mag_mean = torch.sqrt(u_mean * u_mean + v_mean * v_mean)

    eps = 1e-6
    cos = (u * u_mean + v * v_mean) / (mag * mag_mean + eps)
    cos = cos.clamp(-1.0, 1.0)

    gate = torch.sigmoid((cos - float(coh_tau)) * float(coh_k))
    saliency = _compute_attn_mag(flow, q=q, gamma=mag_gamma, t0=mag_t0).unsqueeze(0)
    return (saliency * gate).squeeze(0).clamp(0.0, 1.0)


def _flow_to_uint8(
    flow: torch.Tensor,
    q: float = 0.99,
    *,
    attn_mode: str = "mag",
    coh_tau: float = 0.2,
    coh_k: float = 5.0,
    attn_q: float | None = None,
    attn_gamma: float = 1.0,
    attn_t0: float = 0.0,
    attn_blur: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # flow: [2, H, W] on CPU
    u = flow[0]
    v = flow[1]

    max_abs = max(_quantile_or_max(u.abs(), q), _quantile_or_max(v.abs(), q), 1e-6)

    u01 = (u / max_abs).clamp(-1.0, 1.0) * 0.5 + 0.5
    v01 = (v / max_abs).clamp(-1.0, 1.0) * 0.5 + 0.5

    u8 = (u01 * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
    v8 = (v01 * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

    q_attn = float(attn_q) if attn_q is not None else float(q)

    if attn_mode == "mag":
        attn_t = _compute_attn_mag(flow, q=q_attn, gamma=attn_gamma, t0=attn_t0)
    elif attn_mode == "coh":
        attn_t = _compute_attn_coherence_gate(
            flow,
            q=q_attn,
            coh_tau=coh_tau,
            coh_k=coh_k,
            mag_gamma=attn_gamma,
            mag_t0=attn_t0,
        )
    else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    k = int(attn_blur)
    if k > 0:
        if k % 2 == 0:
            k += 1
        attn_t = F.avg_pool2d(attn_t.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0).squeeze(0)
        attn_t = attn_t.clamp(0.0, 1.0)

    attn = attn_t.cpu().numpy().astype(np.float32)

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


def _merge_no_attn(gray_u8: np.ndarray, u8: np.ndarray, v8: np.ndarray) -> np.ndarray:
    out = np.stack(
        [
            gray_u8.astype(np.uint8),
            v8.astype(np.uint8),
            u8.astype(np.uint8),
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
    ap.add_argument("--attn_q", type=float, default=None)
    ap.add_argument("--attn_gamma", type=float, default=1.0)
    ap.add_argument("--attn_t0", type=float, default=0.0)
    ap.add_argument("--attn_blur", type=int, default=0)
    ap.add_argument("--no_attn", action="store_true")
    ap.add_argument("--normalize_dt", action="store_true")
    ap.add_argument("--attn_mode", type=str, default="mag", choices=["mag", "coh"])
    ap.add_argument("--coh_tau", type=float, default=0.2)
    ap.add_argument("--coh_k", type=float, default=5.0)
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
                prev_idx: int | None = None

                for cur_idx, cur_img_p in frames:
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
                        if args.no_attn:
                            merged = _merge_no_attn(gray_u8, u8, v8)
                        else:
                            merged = _apply_attn(gray_u8, u8, v8, attn, float(args.alpha))
                        Image.fromarray(merged).save(out_img_p)
                        prev_img_p = cur_img_p
                        prev_idx = cur_idx
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

                    if args.normalize_dt:
                        dt = 1
                        if prev_idx is not None:
                            dt = int(cur_idx) - int(prev_idx)
                        if dt <= 0:
                            dt = 1
                        flow = flow / float(dt)

                    u8, v8, attn = _flow_to_uint8(
                        flow,
                        q=float(args.flow_q),
                        attn_mode=str(args.attn_mode),
                        coh_tau=float(args.coh_tau),
                        coh_k=float(args.coh_k),
                        attn_q=args.attn_q,
                        attn_gamma=float(args.attn_gamma),
                        attn_t0=float(args.attn_t0),
                        attn_blur=int(args.attn_blur),
                    )
                    if args.no_attn:
                        merged = _merge_no_attn(gray_u8, u8, v8)
                    else:
                        merged = _apply_attn(gray_u8, u8, v8, attn, float(args.alpha))
                    Image.fromarray(merged).save(out_img_p)

                    prev_img_p = cur_img_p
                    prev_idx = cur_idx

    _write_dataset_yaml(out_root, ["uav"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
