#!/usr/bin/env python3
"""
VisDrone-VID Validation for RAFT+Attn+TNF
Run on AutoDL: python run_visdrone_vid.py

This script:
1. Downloads VisDrone-VID dataset
2. Converts to YOLO format
3. Generates RAFT optical flow with TNF
4. Trains and evaluates with/without TNF for comparison
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm


def download_file(url: str, dest: Path, desc: str = None) -> None:
    """Download a file with progress bar."""
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {desc or url}...")
    
    # Use wget for faster download
    try:
        subprocess.run(["wget", "-O", str(dest), url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to urllib
        urllib.request.urlretrieve(url, dest)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)


def download_visdrone_vid(data_root: Path) -> None:
    """Download VisDrone-VID dataset."""
    # VisDrone-VID URLs (official GitHub releases)
    urls = {
        "train": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-train.zip",
        "val": "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-val.zip",
    }
    
    for split, url in urls.items():
        zip_path = data_root / f"VisDrone2019-VID-{split}.zip"
        download_file(url, zip_path, f"VisDrone-VID {split}")
        
        extract_dir = data_root / f"VisDrone2019-VID-{split}"
        if not extract_dir.exists():
            extract_zip(zip_path, data_root)


def parse_visdrone_annotation(ann_path: Path) -> list:
    """Parse VisDrone annotation file.
    Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    Categories: 0-ignored, 1-pedestrian, 2-people, 3-bicycle, 4-car, 5-van, 6-truck, 7-tricycle, 8-awning-tricycle, 9-bus, 10-motor, 11-others
    """
    if not ann_path.exists():
        return []
    
    boxes = []
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            
            x, y, w, h = map(float, parts[:4])
            score = int(parts[4])
            category = int(parts[5])
            
            # Skip ignored regions (category 0) and others (category 11)
            if category in [0, 11]:
                continue
            
            # Skip zero-size boxes
            if w <= 0 or h <= 0:
                continue
            
            boxes.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'category': category - 1,  # 0-indexed (0-9 for 10 classes)
            })
    
    return boxes


def convert_to_yolo_format(data_root: Path, output_root: Path, sample_rate: int = 5) -> None:
    """Convert VisDrone-VID to YOLO format with frame sampling."""
    # VisDrone classes (excluding ignored and others)
    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    for split in ['train', 'val']:
        src_dir = data_root / f"VisDrone2019-VID-{split}"
        
        # Handle nested directory structure
        if (src_dir / "sequences").exists():
            seq_dir = src_dir / "sequences"
            ann_dir = src_dir / "annotations"
        else:
            # Try alternative structure
            seq_dir = src_dir
            ann_dir = src_dir
        
        if not seq_dir.exists():
            print(f"Warning: {seq_dir} not found")
            continue
        
        out_images = output_root / "images" / split
        out_labels = output_root / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        
        sequences = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
        print(f"Processing {split}: {len(sequences)} sequences")
        
        for seq in tqdm(sequences, desc=f"Converting {split}"):
            # Find images
            img_files = sorted(seq.glob("*.jpg"))
            if not img_files:
                img_files = sorted(seq.glob("*.png"))
            
            # Find annotation file
            ann_file = ann_dir / f"{seq.name}.txt"
            if not ann_file.exists():
                ann_file = seq.parent.parent / "annotations" / f"{seq.name}.txt"
            
            # Parse all annotations
            all_boxes = {}
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 8:
                            continue
                        
                        frame_id = int(parts[0]) if len(parts) > 8 else 1
                        x, y, w, h = map(float, parts[1:5]) if len(parts) > 8 else map(float, parts[:4])
                        category = int(parts[6]) if len(parts) > 8 else int(parts[5])
                        
                        if category in [0, 11] or w <= 0 or h <= 0:
                            continue
                        
                        if frame_id not in all_boxes:
                            all_boxes[frame_id] = []
                        all_boxes[frame_id].append({
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'category': min(category - 1, 9)
                        })
            
            # Sample frames
            for i, img_path in enumerate(img_files):
                if i % sample_rate != 0:
                    continue
                
                frame_idx = i + 1
                
                # Read image size
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
                    continue
                
                # Output paths
                out_name = f"{seq.name}__{img_path.stem}"
                out_img = out_images / f"{out_name}.jpg"
                out_lbl = out_labels / f"{out_name}.txt"
                
                # Copy/link image
                if not out_img.exists():
                    shutil.copy2(img_path, out_img)
                
                # Write YOLO label
                boxes = all_boxes.get(frame_idx, [])
                with open(out_lbl, 'w') as f:
                    for box in boxes:
                        # Convert to YOLO format (normalized xywh)
                        cx = (box['x'] + box['w'] / 2) / img_w
                        cy = (box['y'] + box['h'] / 2) / img_h
                        nw = box['w'] / img_w
                        nh = box['h'] / img_h
                        
                        # Clip to [0, 1]
                        cx = max(0, min(1, cx))
                        cy = max(0, min(1, cy))
                        nw = max(0, min(1, nw))
                        nh = max(0, min(1, nh))
                        
                        f.write(f"{box['category']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    # Write data.yaml
    yaml_content = f"""path: {output_root.as_posix()}
train: images/train
val: images/val
nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
"""
    (output_root / "data.yaml").write_text(yaml_content)
    print(f"YOLO dataset created at: {output_root}")


def generate_raft_tnf(input_root: Path, output_root: Path, normalize_dt: bool = True) -> None:
    """Generate RAFT optical flow with optional TNF."""
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.transforms.functional import to_tensor
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Raft_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = raft_large(weights=weights).to(device).eval()
    
    for split in ["train", "val"]:
        in_images = input_root / "images" / split
        in_labels = input_root / "labels" / split
        out_images = output_root / "images" / split
        out_labels = output_root / "labels" / split
        
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        
        img_paths = sorted(in_images.glob("*.jpg"))
        
        # Group by sequence
        seq_to_frames = {}
        for p in img_paths:
            if "__" in p.stem:
                seq, frame = p.stem.rsplit("__", 1)
                seq_to_frames.setdefault(seq, []).append((frame, p))
        
        for seq in seq_to_frames:
            seq_to_frames[seq].sort(key=lambda x: x[0])
        
        with torch.inference_mode():
            for seq, frames in tqdm(seq_to_frames.items(), desc=f"RAFT {split}"):
                prev_img = None
                prev_idx = None
                
                for frame_name, cur_path in frames:
                    out_img_path = out_images / cur_path.name
                    out_lbl_path = out_labels / f"{cur_path.stem}.txt"
                    
                    # Copy label
                    src_lbl = in_labels / f"{cur_path.stem}.txt"
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, out_lbl_path)
                    else:
                        out_lbl_path.write_text("")
                    
                    cur_img = Image.open(cur_path).convert("RGB")
                    gray = np.array(cur_img.convert("L"), dtype=np.uint8)
                    
                    try:
                        cur_idx = int(frame_name)
                    except:
                        cur_idx = 0
                    
                    if prev_img is None:
                        # No flow for first frame
                        u8 = np.full_like(gray, 128)
                        v8 = np.full_like(gray, 128)
                        attn = np.zeros_like(gray, dtype=np.float32)
                    else:
                        t1 = to_tensor(prev_img)
                        t2 = to_tensor(cur_img)
                        t1, t2 = preprocess(t1, t2)
                        
                        pred = model(t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device))
                        flow = pred[-1][0].cpu().float()
                        
                        # Resize if needed
                        h0, w0 = gray.shape
                        if flow.shape[-2:] != (h0, w0):
                            flow = F.interpolate(flow.unsqueeze(0), size=(h0, w0), mode="bilinear")[0]
                        
                        # TNF: normalize by dt
                        if normalize_dt and prev_idx is not None:
                            dt = max(1, cur_idx - prev_idx)
                            flow = flow / float(dt)
                        
                        # Convert to uint8
                        u, v = flow[0], flow[1]
                        max_abs = max(float(torch.quantile(u.abs().flatten(), 0.99)),
                                     float(torch.quantile(v.abs().flatten(), 0.99)), 1e-6)
                        
                        u01 = (u / max_abs).clamp(-1, 1) * 0.5 + 0.5
                        v01 = (v / max_abs).clamp(-1, 1) * 0.5 + 0.5
                        
                        u8 = (u01 * 255).round().clamp(0, 255).to(torch.uint8).numpy()
                        v8 = (v01 * 255).round().clamp(0, 255).to(torch.uint8).numpy()
                        
                        # Magnitude attention
                        mag = torch.sqrt(u*u + v*v)
                        mag_scale = max(float(torch.quantile(mag.flatten(), 0.99)), 1e-6)
                        attn = (mag / mag_scale).clamp(0, 1).pow(0.5).numpy()
                    
                    # Merge: Gray + V + U (with attention modulation)
                    gain = 1.0 + attn
                    g_out = np.clip(gray.astype(np.float32) / 255 * gain, 0, 1)
                    u_out = np.clip(u8.astype(np.float32) / 255 * attn, 0, 1)
                    v_out = np.clip(v8.astype(np.float32) / 255 * attn, 0, 1)
                    
                    merged = np.stack([
                        (g_out * 255).astype(np.uint8),
                        (v_out * 255).astype(np.uint8),
                        (u_out * 255).astype(np.uint8),
                    ], axis=-1)
                    
                    Image.fromarray(merged).save(out_img_path)
                    
                    prev_img = cur_img
                    prev_idx = cur_idx
    
    # Copy data.yaml
    yaml_src = input_root / "data.yaml"
    yaml_dst = output_root / "data.yaml"
    if yaml_src.exists():
        content = yaml_src.read_text().replace(str(input_root), str(output_root))
        yaml_dst.write_text(content)
    
    print(f"RAFT+TNF data generated at: {output_root}")


def train_yolo(data_yaml: Path, run_name: str, epochs: int = 100) -> dict:
    """Train YOLOv8 and return results."""
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=32,
        device=0,
        project="runs/visdrone",
        name=run_name,
        exist_ok=True,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        verbose=True,
    )
    
    return {
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="VisDrone-VID validation for RAFT+Attn+TNF")
    parser.add_argument("--data_root", type=str, default="Data/VisDrone")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--skip_raft", action="store_true")
    parser.add_argument("--sample_rate", type=int, default=5, help="Sample every N frames")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    data_root = project_root / args.data_root
    yolo_root = data_root / "yolo"
    raft_tnf_root = data_root / "yolo_raft_tnf"
    raft_no_tnf_root = data_root / "yolo_raft_no_tnf"
    
    # Step 1: Download
    if not args.skip_download:
        print("\n=== Step 1: Download VisDrone-VID ===")
        download_visdrone_vid(data_root)
    
    # Step 2: Convert to YOLO format
    if not args.skip_convert:
        print("\n=== Step 2: Convert to YOLO format ===")
        convert_to_yolo_format(data_root, yolo_root, sample_rate=args.sample_rate)
    
    # Step 3: Generate RAFT + TNF
    if not args.skip_raft:
        print("\n=== Step 3: Generate RAFT + Attn + TNF ===")
        generate_raft_tnf(yolo_root, raft_tnf_root, normalize_dt=True)
        
        print("\n=== Step 3b: Generate RAFT + Attn (no TNF) for comparison ===")
        generate_raft_tnf(yolo_root, raft_no_tnf_root, normalize_dt=False)
    
    # Step 4: Train and compare
    print("\n=== Step 4: Training ===")
    results = {}
    
    # RGB baseline
    print("\n--- Training RGB baseline ---")
    results["rgb"] = train_yolo(yolo_root / "data.yaml", "rgb_baseline", args.epochs)
    
    # RAFT + Attn + TNF
    print("\n--- Training RAFT + Attn + TNF ---")
    results["raft_tnf"] = train_yolo(raft_tnf_root / "data.yaml", "raft_attn_tnf", args.epochs)
    
    # RAFT + Attn (no TNF)
    print("\n--- Training RAFT + Attn (no TNF) ---")
    results["raft_no_tnf"] = train_yolo(raft_no_tnf_root / "data.yaml", "raft_attn_no_tnf", args.epochs)
    
    # Print results
    print("\n" + "=" * 80)
    print("VISDRONE-VID RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Method':<25} {'mAP50':<10} {'mAP50-95':<10} {'Recall':<10} {'Precision':<10}")
    print("-" * 65)
    
    for name, r in results.items():
        print(f"{name:<25} {r['mAP50']:.4f}    {r['mAP50-95']:.4f}    {r['recall']:.4f}    {r['precision']:.4f}")
    
    # Save results
    out_file = project_root / "visdrone_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_file}")
    
    # Highlight TNF improvement
    if "raft_tnf" in results and "raft_no_tnf" in results:
        tnf_gain = results["raft_tnf"]["mAP50"] - results["raft_no_tnf"]["mAP50"]
        print(f"\n*** TNF improvement: +{tnf_gain*100:.2f}% mAP50 ***")


if __name__ == "__main__":
    main()
