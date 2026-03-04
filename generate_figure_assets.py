"""
生成框架图所需的素材图片
在 AutoDL 上运行：
    cd /root/proj1
    python generate_figure_assets.py --in_root Data/yolo_fold4 --out_dir figure_assets
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import to_tensor
import cv2


def flow_to_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """将光流转换为HSV彩色可视化"""
    h, w = u.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi) * 179  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)  # Value
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_attention_mag(flow: torch.Tensor, gamma: float = 0.5) -> np.ndarray:
    """计算基于magnitude的注意力图"""
    u = flow[0].numpy()
    v = flow[1].numpy()
    mag = np.sqrt(u**2 + v**2)
    mag_norm = mag / (np.percentile(mag, 99) + 1e-6)
    mag_norm = np.clip(mag_norm, 0, 1)
    attn = np.power(mag_norm, gamma)
    return attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="Data/yolo_fold4", help="输入数据目录")
    ap.add_argument("--out_dir", type=str, default="figure_assets", help="输出素材目录")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seq", type=str, default=None, help="指定序列名，否则自动选择第一个")
    ap.add_argument("--frame_idx", type=int, default=10, help="选择第几帧作为示例")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 RAFT 模型
    weights = Raft_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = raft_large(weights=weights, progress=True).to(device).eval()

    # 找到图像
    in_images = in_root / "images" / "train"
    img_paths = sorted([p for p in in_images.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    
    # 解析序列
    seq_to_frames = {}
    for p in img_paths:
        stem = p.stem
        if "__" not in stem:
            continue
        seq, frame = stem.split("__", 1)
        try:
            idx = int(frame)
            seq_to_frames.setdefault(seq, []).append((idx, p))
        except ValueError:
            continue
    
    for seq in seq_to_frames:
        seq_to_frames[seq].sort(key=lambda x: x[0])
    
    # 选择序列
    if args.seq and args.seq in seq_to_frames:
        selected_seq = args.seq
    else:
        selected_seq = list(seq_to_frames.keys())[0]
    
    frames = seq_to_frames[selected_seq]
    print(f"Selected sequence: {selected_seq}, {len(frames)} frames")
    
    # 选择帧
    frame_idx = min(args.frame_idx, len(frames) - 1)
    frame_idx = max(frame_idx, 1)  # 至少从第1帧开始（需要前一帧）
    
    idx_t, path_t = frames[frame_idx]
    idx_t_prev, path_t_prev = frames[frame_idx - 1]
    idx_t_next, path_t_next = frames[min(frame_idx + 1, len(frames) - 1)]
    
    print(f"Using frames: t-1={idx_t_prev}, t={idx_t}, t+1={idx_t_next}")

    # 1. 保存 RGB 帧
    img_t_prev = Image.open(path_t_prev).convert("RGB")
    img_t = Image.open(path_t).convert("RGB")
    img_t_next = Image.open(path_t_next).convert("RGB")
    
    img_t_prev.save(out_dir / "frame_t-1.png")
    img_t.save(out_dir / "frame_t.png")
    img_t_next.save(out_dir / "frame_t+1.png")
    print("Saved: frame_t-1.png, frame_t.png, frame_t+1.png")

    # 2. 计算光流
    with torch.inference_mode():
        t1 = to_tensor(img_t_prev)
        t2 = to_tensor(img_t)
        t1_p, t2_p = preprocess(t1, t2)
        
        pred = model(t1_p.unsqueeze(0).to(device), t2_p.unsqueeze(0).to(device))
        flow = pred[-1][0].detach().float().cpu()  # [2, H, W]
        
        # 调整到原始尺寸
        h0, w0 = img_t.size[1], img_t.size[0]
        h1, w1 = flow.shape[-2], flow.shape[-1]
        if (h1, w1) != (h0, w0):
            flow = F.interpolate(flow.unsqueeze(0), size=(h0, w0), mode="bilinear", align_corners=False)[0]
            flow[0] *= w0 / w1
            flow[1] *= h0 / h1

    u = flow[0].numpy()
    v = flow[1].numpy()

    # 3. 光流可视化 (原始)
    flow_color = flow_to_color(u, v)
    Image.fromarray(flow_color).save(out_dir / "flow_raw.png")
    print("Saved: flow_raw.png")

    # 4. 有噪声的光流 (就是原始光流)
    Image.fromarray(flow_color).save(out_dir / "flow_noisy.png")
    print("Saved: flow_noisy.png")

    # 5. 注意力图
    attn = compute_attention_mag(flow, gamma=0.5)
    attn_u8 = (attn * 255).astype(np.uint8)
    Image.fromarray(attn_u8).save(out_dir / "attention_map.png")
    print("Saved: attention_map.png")

    # 6. 去噪后的光流 (用注意力加权)
    u_clean = u * attn
    v_clean = v * attn
    flow_clean_color = flow_to_color(u_clean, v_clean)
    Image.fromarray(flow_clean_color).save(out_dir / "flow_clean.png")
    print("Saved: flow_clean.png")

    # 7. RGB 通道分离
    img_t_np = np.array(img_t)
    Image.fromarray(img_t_np[..., 0]).save(out_dir / "channel_R.png")
    Image.fromarray(img_t_np[..., 1]).save(out_dir / "channel_G.png")
    Image.fromarray(img_t_np[..., 2]).save(out_dir / "channel_B.png")
    print("Saved: channel_R.png, channel_G.png, channel_B.png")

    # 8. 光流通道 (归一化到 0-255)
    # dt-normalization (假设 dt=1)
    dt = idx_t - idx_t_prev
    if dt <= 0:
        dt = 1
    u_norm = u / dt
    v_norm = v / dt
    
    # 归一化到 0-255
    u_scale = np.percentile(np.abs(u_norm), 99) + 1e-6
    v_scale = np.percentile(np.abs(v_norm), 99) + 1e-6
    
    u_u8 = ((u_norm / u_scale * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    v_u8 = ((v_norm / v_scale * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    
    Image.fromarray(u_u8).save(out_dir / "channel_u.png")
    Image.fromarray(v_u8).save(out_dir / "channel_v.png")
    Image.fromarray(attn_u8).save(out_dir / "channel_A.png")
    print("Saved: channel_u.png, channel_v.png, channel_A.png")

    print(f"\n所有素材已保存到: {out_dir.absolute()}")
    print("文件列表:")
    for f in sorted(out_dir.iterdir()):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
