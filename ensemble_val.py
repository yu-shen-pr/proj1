"""
Ensemble validation: RGB + RAFT/TNF models fused via WBF (Weighted Box Fusion).
Works with Ultralytics 8.4.x — no multi-weight AutoBackend needed.

Usage:
    python ensemble_val.py \
        --rgb   /workspace/proj1/runs/5fold_cv/rgb_no_hsv_fold4/weights/best.pt \
        --tnf   /workspace/proj1/runs/5fold_cv/raft_tnf_fold4_motionaug/weights/best.pt \
        --data  /workspace/proj1/Data/yolo_fold4/data.yaml \
        --iou   0.55 --conf 0.001 --imgsz 640 --device 0
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset


# ---------- helpers ----------

def load_gt(label_dir: str, img_ids: list[str], nc: int):
    """Return list of (N,5) arrays [cls,x,y,w,h] per image (YOLO format, normalised)."""
    gts = []
    for img_id in img_ids:
        p = Path(label_dir) / (Path(img_id).stem + ".txt")
        if p.exists():
            data = np.loadtxt(p, ndmin=2).astype(np.float32)
            gts.append(data)          # shape (N,5)
        else:
            gts.append(np.zeros((0, 5), np.float32))
    return gts


def wbf_single(boxes_list, scores_list, iou_thr: float = 0.55, skip_box_thr: float = 0.001):
    """
    Simple WBF (Weighted Box Fusion) for a single image.
    boxes_list : list of (N,4) np arrays in xyxy [0..1] normalised
    scores_list: list of (N,) confidence scores
    Returns fused (M,4) boxes and (M,) scores.
    """
    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores in zip(boxes_list, scores_list):
        for b, s in zip(boxes, scores):
            if s >= skip_box_thr:
                all_boxes.append(b)
                all_scores.append(s)
                all_labels.append(0)

    if len(all_boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    all_boxes  = np.array(all_boxes,  np.float32)   # (M,4)
    all_scores = np.array(all_scores, np.float32)   # (M,)

    # greedy cluster by IoU
    order = np.argsort(-all_scores)
    used  = np.zeros(len(order), bool)
    clusters_boxes, clusters_scores = [], []

    for idx in order:
        if used[idx]:
            continue
        b0 = all_boxes[idx]
        cluster_b = [b0]
        cluster_s = [all_scores[idx]]
        used[idx] = True
        for jdx in order:
            if used[jdx]:
                continue
            bj = all_boxes[jdx]
            iou = _iou(b0, bj)
            if iou >= iou_thr:
                cluster_b.append(bj)
                cluster_s.append(all_scores[jdx])
                used[jdx] = True
        cb = np.array(cluster_b)
        cs = np.array(cluster_s)
        w  = cs / cs.sum()
        fused_box   = (w[:, None] * cb).sum(axis=0)
        fused_score = cs.mean()
        clusters_boxes.append(fused_box)
        clusters_scores.append(fused_score)

    return np.array(clusters_boxes, np.float32), np.array(clusters_scores, np.float32)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-7)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb",      required=True)
    ap.add_argument("--tnf",      required=True)
    ap.add_argument("--data",     required=True,  help="RGB dataset yaml (also used for GT labels)")
    ap.add_argument("--data_tnf", default=None,   help="TNF/motion dataset yaml (if different from --data)")
    ap.add_argument("--iou",      type=float, default=0.55)
    ap.add_argument("--conf",     type=float, default=0.001)
    ap.add_argument("--imgsz",    type=int,   default=640)
    ap.add_argument("--device",   default="0")
    ap.add_argument("--tta",      action="store_true")
    args = ap.parse_args()

    # load RGB dataset yaml -> GT labels + RGB image paths
    with open(args.data) as f:
        ds_cfg = yaml.safe_load(f)
    ds_root  = Path(args.data).parent
    val_img  = ds_root / ds_cfg.get("val", "images/val")
    val_lbl  = ds_root / ds_cfg.get("val", "images/val").replace("images", "labels")
    nc       = int(ds_cfg.get("nc", 1))

    rgb_img_paths = sorted(
        [str(p) for p in val_img.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    )

    # load TNF dataset yaml -> motion image paths (same filenames, different root)
    if args.data_tnf:
        with open(args.data_tnf) as f:
            tnf_cfg = yaml.safe_load(f)
        tnf_root    = Path(args.data_tnf).parent
        tnf_val_img = tnf_root / tnf_cfg.get("val", "images/val")
        tnf_img_paths = sorted(
            [str(p) for p in tnf_val_img.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
        )
        # align by stem name
        rgb_stem_to_path = {Path(p).stem: p for p in rgb_img_paths}
        tnf_stem_to_path = {Path(p).stem: p for p in tnf_img_paths}
        common_stems = sorted(set(rgb_stem_to_path) & set(tnf_stem_to_path))
        rgb_img_paths = [rgb_stem_to_path[s] for s in common_stems]
        tnf_img_paths_aligned = [tnf_stem_to_path[s] for s in common_stems]
        print(f"Val images: {len(rgb_img_paths)} (RGB) / {len(tnf_img_paths_aligned)} (TNF),  nc={nc}")
    else:
        tnf_img_paths_aligned = rgb_img_paths
        print(f"Val images: {len(rgb_img_paths)},  nc={nc}  [WARNING: TNF model using RGB images]")

    img_paths = rgb_img_paths  # used for GT label lookup
    # O(1) lookup: rgb_path -> tnf_path
    rgb_to_tnf = {rgb: tnf for rgb, tnf in zip(rgb_img_paths, tnf_img_paths_aligned)}

    # load models
    print("Loading RGB model …")
    rgb_model = YOLO(args.rgb)
    print("Loading TNF model …")
    tnf_model = YOLO(args.tnf)

    # collect stats
    stats   = []   # list of (correct, conf, pred_cls, target_cls)
    seen    = 0
    iouv    = torch.linspace(0.5, 0.95, 10)   # for mAP50-95
    niou    = len(iouv)

    for img_path in tqdm(img_paths, desc="Validating"):
        # ---- ground truth ----
        lbl_path = Path(val_lbl) / (Path(img_path).stem + ".txt")
        if lbl_path.exists():
            gt_raw = np.loadtxt(str(lbl_path), ndmin=2).astype(np.float32)
        else:
            gt_raw = np.zeros((0, 5), np.float32)

        # ---- predictions from each model ----
        def get_preds(model, path):
            res = model.predict(
                path, imgsz=args.imgsz, conf=args.conf, iou=0.99,   # very loose NMS
                augment=args.tta, device=args.device, verbose=False
            )
            r = res[0]
            if r.boxes is None or len(r.boxes) == 0:
                return np.zeros((0,4),np.float32), np.zeros(0,np.float32)
            boxes_xyxy = r.boxes.xyxyn.cpu().numpy()  # normalised xyxy
            confs      = r.boxes.conf.cpu().numpy()
            return boxes_xyxy, confs

        tnf_img_path = rgb_to_tnf[img_path]
        b1, s1 = get_preds(rgb_model, img_path)
        b2, s2 = get_preds(tnf_model, tnf_img_path)

        # ---- WBF fusion ----
        fb, fs = wbf_single([b1, b2], [s1, s2], iou_thr=args.iou)

        # ---- eval: match preds to GT ----
        seen += 1
        ngt = len(gt_raw)

        if len(fb) == 0:
            if ngt > 0:
                stats.append((
                    torch.zeros(0, niou, dtype=torch.bool),
                    torch.zeros(0),
                    torch.zeros(0, dtype=torch.int),
                    torch.zeros(ngt, dtype=torch.int)
                ))
            continue

        pred_xyxy = torch.from_numpy(fb).float()    # (M,4)
        pred_conf = torch.from_numpy(fs).float()    # (M,)
        pred_cls  = torch.zeros(len(fb), dtype=torch.int)  # single class

        if ngt == 0:
            stats.append((
                torch.zeros(len(fb), niou, dtype=torch.bool),
                pred_conf,
                pred_cls,
                torch.zeros(0, dtype=torch.int)
            ))
            continue

        gt_cls    = torch.from_numpy(gt_raw[:, 0]).int()
        # convert gt xywh norm -> xyxy norm (need image size from first predict)
        # we already have val image; get h,w via PIL
        from PIL import Image
        wh = Image.open(img_path).size   # (W, H)
        W, H = wh
        gt_xyxy = torch.from_numpy(gt_raw[:, 1:]).float()
        gt_xyxy[:, 0] = (gt_raw[:, 1] - gt_raw[:, 3]/2) * W
        gt_xyxy[:, 1] = (gt_raw[:, 2] - gt_raw[:, 4]/2) * H
        gt_xyxy[:, 2] = (gt_raw[:, 1] + gt_raw[:, 3]/2) * W
        gt_xyxy[:, 3] = (gt_raw[:, 2] + gt_raw[:, 4]/2) * H
        # pred is normalised, scale to pixels too
        pred_pix = pred_xyxy.clone()
        pred_pix[:, [0,2]] *= W
        pred_pix[:, [1,3]] *= H

        correct = _match_predictions(pred_pix, gt_xyxy, iouv)
        stats.append((correct, pred_conf, pred_cls, gt_cls))

    # ---- aggregate metrics ----
    stats_cat = [torch.cat(x, 0) if len(x) else torch.zeros(0) for x in zip(*stats)]
    if len(stats_cat) and stats_cat[0].numel():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            stats_cat[0].numpy(),
            stats_cat[1].numpy(),
            stats_cat[2].numpy(),
            stats_cat[3].numpy(),
            plot=False,
            names={0: "target"}
        )
        map50    = ap[:, 0].mean()
        map5095  = ap.mean()
        p_mean   = p.mean()
        r_mean   = r.mean()
    else:
        p_mean = r_mean = map50 = map5095 = 0.0

    tag = "Ensemble+TTA" if args.tta else "Ensemble"
    print(f"\n{tag}  P={p_mean:.4f}  R={r_mean:.4f}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")


def _match_predictions(pred_xyxy, gt_xyxy, iouv):
    """Returns (M, niou) bool tensor: correct[i,j] = pred i matched at IoU threshold iouv[j]."""
    niou = len(iouv)
    correct = torch.zeros(len(pred_xyxy), niou, dtype=torch.bool)
    if len(gt_xyxy) == 0:
        return correct
    # IoU matrix (M x G)
    iou_mat = box_iou(pred_xyxy, gt_xyxy)   # (M, G)
    for j, thr in enumerate(iouv):
        matches = (iou_mat >= thr).nonzero(as_tuple=False)
        if matches.numel():
            # each gt matched at most once, greedy by IoU
            iou_vals = iou_mat[matches[:, 0], matches[:, 1]]
            order    = iou_vals.argsort(descending=True)
            matches  = matches[order]
            _, ui    = torch.unique(matches[:, 1], return_inverse=True)
            first    = torch.zeros(matches.shape[0], dtype=torch.bool)
            seen_g   = set()
            seen_p   = set()
            for k in range(len(matches)):
                g = matches[k, 1].item()
                p = matches[k, 0].item()
                if g not in seen_g and p not in seen_p:
                    first[k] = True
                    seen_g.add(g)
                    seen_p.add(p)
            correct[matches[first, 0], j] = True
    return correct


def box_iou(box1, box2):
    """Compute IoU between two sets of boxes (xyxy)."""
    A = box1.shape[0]; B = box2.shape[0]
    b1 = box1.unsqueeze(1).expand(A, B, 4)
    b2 = box2.unsqueeze(0).expand(A, B, 4)
    ix1 = torch.max(b1[...,0], b2[...,0])
    iy1 = torch.max(b1[...,1], b2[...,1])
    ix2 = torch.min(b1[...,2], b2[...,2])
    iy2 = torch.min(b1[...,3], b2[...,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    a1 = (box1[:,2]-box1[:,0])*(box1[:,3]-box1[:,1])
    a2 = (box2[:,2]-box2[:,0])*(box2[:,3]-box2[:,1])
    union = a1.unsqueeze(1) + a2.unsqueeze(0) - inter
    return inter / (union + 1e-7)


if __name__ == "__main__":
    main()
