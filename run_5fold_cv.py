#!/usr/bin/env python3
"""
5-Fold Cross-Validation for RAFT+Attn+TNF
Run on AutoDL: python run_5fold_cv.py

This script:
1. Generates YOLO format datasets for each fold
2. Generates RAFT optical flow with TNF (Temporal-Normalized Flow) and attention
3. Trains YOLOv8 on each fold
4. Collects and reports results with mean ± std
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import yaml


def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a command and return exit code."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def generate_fold_lists(project_root: Path, best_folds_dir: Path) -> None:
    """Generate newdata.txt files for all 5 folds based on sequence splits."""
    data_train = project_root / "Data" / "train"
    
    for fold in range(1, 6):
        train_seq_file = best_folds_dir / f"train_full_fold_{fold}.txt"
        valid_seq_file = best_folds_dir / f"valid_full_fold_{fold}.txt"
        
        if not train_seq_file.exists() or not valid_seq_file.exists():
            print(f"Fold {fold}: Missing sequence files, skipping generation")
            continue
        
        # Read sequence names
        train_seqs = [l.strip() for l in train_seq_file.read_text().splitlines() if l.strip()]
        valid_seqs = [l.strip() for l in valid_seq_file.read_text().splitlines() if l.strip()]
        
        train_imgs = []
        valid_imgs = []
        
        # Collect images from each sequence (every 5 frames)
        for seq in train_seqs:
            seq_dir = data_train / seq
            if not seq_dir.exists():
                print(f"Warning: sequence {seq} not found")
                continue
            imgs = sorted(seq_dir.glob("*.jpg"))
            # Sample every 5 frames
            for i, img in enumerate(imgs):
                if i % 5 == 0:
                    rel_path = f"./Data/train/{seq}/{img.name}"
                    train_imgs.append(rel_path)
        
        for seq in valid_seqs:
            seq_dir = data_train / seq
            if not seq_dir.exists():
                print(f"Warning: sequence {seq} not found")
                continue
            imgs = sorted(seq_dir.glob("*.jpg"))
            for i, img in enumerate(imgs):
                if i % 5 == 0:
                    rel_path = f"./Data/train/{seq}/{img.name}"
                    valid_imgs.append(rel_path)
        
        # Write newdata files
        out_train = best_folds_dir / f"train_fold{fold}_newdata.txt"
        out_valid = best_folds_dir / f"valid_fold{fold}_newdata.txt"
        
        if not out_train.exists() or out_train.stat().st_size < 1000:
            out_train.write_text("\n".join(train_imgs) + "\n", encoding="utf-8")
            print(f"Generated {out_train} with {len(train_imgs)} images")
        
        if not out_valid.exists() or out_valid.stat().st_size < 1000:
            out_valid.write_text("\n".join(valid_imgs) + "\n", encoding="utf-8")
            print(f"Generated {out_valid} with {len(valid_imgs)} images")


def prepare_fold_data(fold: int, project_root: Path) -> Path:
    """Prepare YOLO + RAFT+Attn+TNF data for a specific fold."""
    yolo_dir = project_root / "Data" / f"yolo_fold{fold}"
    raft_dir = project_root / "Data" / f"yolo_fold{fold}_raft_attn_tnf"
    
    # Step 1: Convert fold to YOLO format
    if not yolo_dir.exists() or not (yolo_dir / "data.yaml").exists():
        print(f"\n=== Fold {fold}: Converting to YOLO format ===")
        run_cmd([
            sys.executable, "Data/convert_fold_to_yolo.py",
            "--fold", str(fold),
            "--out", f"Data/yolo_fold{fold}",
            "--mode", "copy"
        ], cwd=project_root)
    else:
        print(f"Fold {fold}: YOLO data already exists at {yolo_dir}")
    
    # Step 2: Generate RAFT + Attention + TNF
    if not raft_dir.exists() or not (raft_dir / "data.yaml").exists():
        print(f"\n=== Fold {fold}: Generating RAFT+Attn+TNF ===")
        run_cmd([
            sys.executable, "Data/gen_raft_flow_attn.py",
            "--in_root", f"Data/yolo_fold{fold}",
            "--out_root", f"Data/yolo_fold{fold}_raft_attn_tnf",
            "--normalize_dt",  # TNF: Temporal-Normalized Flow
            "--attn_mode", "mag",
            "--attn_gamma", "0.5",
            "--attn_blur", "3",
            "--device", "cuda"
        ], cwd=project_root)
    else:
        print(f"Fold {fold}: RAFT+TNF data already exists at {raft_dir}")
    
    return raft_dir


def train_fold(fold: int, project_root: Path, data_dir: Path, epochs: int = 300) -> dict:
    """Train YOLOv8 on a specific fold and return results."""
    from ultralytics import YOLO
    
    run_name = f"raft_attn_tnf_fold{fold}_e{epochs}"
    data_yaml = data_dir / "data.yaml"
    
    print(f"\n=== Fold {fold}: Training YOLOv8 for {epochs} epochs ===")
    print(f"Data: {data_yaml}")
    print(f"Run name: {run_name}")
    
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=(640, 512),
        batch=32,
        device=0,
        project=str(project_root / "runs" / "5fold_cv"),
        name=run_name,
        exist_ok=True,
        hsv_h=0.0,  # Disable HSV augmentation for motion channels
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        verbose=True,
    )
    
    # Get best metrics from validation
    best_results = {
        "fold": fold,
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
    }
    
    return best_results


def validate_fold(fold: int, project_root: Path, data_dir: Path) -> dict:
    """Validate a trained model on its fold."""
    from ultralytics import YOLO
    
    # Find best model
    run_dir = project_root / "runs" / "5fold_cv" / f"raft_attn_tnf_fold{fold}_e300"
    best_pt = run_dir / "weights" / "best.pt"
    
    if not best_pt.exists():
        print(f"Warning: best.pt not found for fold {fold}")
        return {}
    
    data_yaml = data_dir / "data.yaml"
    
    model = YOLO(str(best_pt))
    metrics = model.val(data=str(data_yaml), imgsz=(640, 512), device=0)
    
    return {
        "fold": fold,
        "precision": float(metrics.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(metrics.results_dict.get("metrics/recall(B)", 0)),
        "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0)),
    }


def compute_statistics(results: list[dict]) -> dict:
    """Compute mean and std for each metric."""
    import numpy as np
    
    metrics = ["precision", "recall", "mAP50", "mAP50-95"]
    stats = {}
    
    for m in metrics:
        values = [r[m] for r in results if m in r]
        if values:
            stats[m] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    
    return stats


def print_results_table(results: list[dict], stats: dict) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION RESULTS: RAFT + Attn + TNF")
    print("=" * 80)
    
    # Per-fold results
    print(f"\n{'Fold':<8} {'Precision':<12} {'Recall':<12} {'mAP50':<12} {'mAP50-95':<12}")
    print("-" * 56)
    
    for r in results:
        print(f"{r['fold']:<8} {r['precision']:.4f}       {r['recall']:.4f}       {r['mAP50']:.4f}       {r['mAP50-95']:.4f}")
    
    print("-" * 56)
    
    # Summary statistics
    print(f"\n{'Metric':<12} {'Mean':<12} {'Std':<12} {'Mean ± Std':<20}")
    print("-" * 56)
    
    for m in ["precision", "recall", "mAP50", "mAP50-95"]:
        if m in stats:
            mean = stats[m]["mean"]
            std = stats[m]["std"]
            print(f"{m:<12} {mean:.4f}       {std:.4f}       {mean:.3f} ± {std:.3f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="5-Fold Cross-Validation for RAFT+Attn+TNF")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="Comma-separated fold numbers")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs per fold")
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing models")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent
    best_folds_dir = project_root / "Data" / "best_folds"
    
    folds = [int(f.strip()) for f in args.folds.split(",")]
    print(f"Running 5-fold CV on folds: {folds}")
    print(f"Epochs per fold: {args.epochs}")
    
    # Step 0: Generate fold list files if needed
    if not args.skip_data_prep:
        print("\n=== Checking fold list files ===")
        generate_fold_lists(project_root, best_folds_dir)
    
    results = []
    
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        
        # Prepare data
        if not args.skip_data_prep:
            data_dir = prepare_fold_data(fold, project_root)
        else:
            data_dir = project_root / "Data" / f"yolo_fold{fold}_raft_attn_tnf"
        
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            continue
        
        # Train or validate
        if args.validate_only:
            fold_results = validate_fold(fold, project_root, data_dir)
        else:
            fold_results = train_fold(fold, project_root, data_dir, epochs=args.epochs)
        
        if fold_results:
            results.append(fold_results)
            print(f"\nFold {fold} results: {fold_results}")
    
    # Compute and print statistics
    if results:
        stats = compute_statistics(results)
        print_results_table(results, stats)
        
        # Save results to JSON
        output = {
            "timestamp": datetime.now().isoformat(),
            "epochs": args.epochs,
            "folds": folds,
            "per_fold_results": results,
            "statistics": stats,
        }
        
        out_file = project_root / "5fold_cv_results.json"
        out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_file}")
        
        # Print LaTeX-ready format
        print("\n=== LaTeX Table Row ===")
        for m in ["mAP50", "recall", "precision", "mAP50-95"]:
            if m in stats:
                print(f"{m}: ${stats[m]['mean']:.3f} \\pm {stats[m]['std']:.3f}$")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
