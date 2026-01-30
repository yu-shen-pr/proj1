import argparse
import os
import sys
from typing import Any


def _parse_imgsz(s: str) -> Any:
    s = str(s).strip()
    if "," in s:
        a, b = s.split(",", 1)
        return [int(a), int(b)]
    return int(s)


def _apply_kv_overrides(ns: argparse.Namespace, argv: list[str]) -> None:
    for a in argv:
        if a.startswith("--"):
            continue
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        if not hasattr(ns, k):
            continue
        cur = getattr(ns, k)
        if isinstance(cur, bool):
            vv = v.lower() in {"1", "true", "yes", "y"}
        elif isinstance(cur, int):
            vv = int(float(v))
        elif isinstance(cur, float):
            vv = float(v)
        else:
            vv = v
        setattr(ns, k, vv)


def _import_yolo_dataset_class():
    try:
        from ultralytics.data.dataset import YOLODataset  # type: ignore

        return YOLODataset
    except Exception:
        from ultralytics.data.dataset import YOLODataset  # type: ignore

        return YOLODataset


def _patch_yolodataset_for_motion(motion_dirname: str = "motion_images") -> None:
    YOLODataset = _import_yolo_dataset_class()

    if getattr(YOLODataset, "_sixch_patched", False):
        return

    orig_load_image = YOLODataset.load_image

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
    YOLODataset._sixch_patched = True  # type: ignore


def _expand_first_conv_inplace(yolo, in_ch: int = 6) -> None:
    import torch
    import torch.nn as nn

    root = yolo.model

    # Prefer the canonical location for Ultralytics DetectionModel: root.model[0] is the first Conv wrapper
    old = None
    wrapper = None
    if hasattr(root, "model"):
        m0 = getattr(root, "model")
        try:
            wrapper = m0[0]
            if hasattr(wrapper, "conv") and isinstance(wrapper.conv, nn.Conv2d):
                old = wrapper.conv
        except Exception:
            old = None

    # Fallback: locate first Conv2d by module traversal
    if old is None:
        for _, m in root.named_modules():
            if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
                old = m
                break

    if old is None:
        raise RuntimeError("Failed to locate the first Conv2d with in_channels=3")

    new = nn.Conv2d(
        in_channels=in_ch,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if in_ch > old.in_channels:
            mean_w = old.weight.mean(dim=1, keepdim=True)
            rep = in_ch - old.in_channels
            new.weight[:, old.in_channels :].copy_(mean_w.repeat(1, rep, 1, 1))
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)

    # Write back
    if wrapper is not None and hasattr(wrapper, "conv"):
        wrapper.conv = new
        if hasattr(wrapper, "c1"):
            wrapper.c1 = in_ch
    else:
        # last-resort: replace by name path
        target_name = None
        for name, m in root.named_modules():
            if m is old:
                target_name = name
                break
        if not target_name:
            raise RuntimeError("Failed to resolve module path for first conv")
        parent = root
        attr = target_name
        if "." in target_name:
            parts = target_name.split(".")
            attr = parts[-1]
            for p in parts[:-1]:
                if p.isdigit():
                    parent = parent[int(p)]
                else:
                    parent = getattr(parent, p)
        setattr(parent, attr, new)

    if hasattr(root, "yaml") and isinstance(root.yaml, dict):
        root.yaml["ch"] = in_ch


def _expand_first_conv_model(model, in_ch: int = 6) -> None:
    import torch
    import torch.nn as nn

    wrapper = None
    old = None
    if hasattr(model, "model"):
        try:
            wrapper = model.model[0]
            if hasattr(wrapper, "conv") and isinstance(wrapper.conv, nn.Conv2d):
                old = wrapper.conv
        except Exception:
            old = None

    if old is None:
        for _, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
                old = m
                break

    if old is None:
        raise RuntimeError("Failed to locate first conv in trainer model")

    new = nn.Conv2d(
        in_channels=in_ch,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if in_ch > old.in_channels:
            mean_w = old.weight.mean(dim=1, keepdim=True)
            rep = in_ch - old.in_channels
            new.weight[:, old.in_channels :].copy_(mean_w.repeat(1, rep, 1, 1))
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)

    if wrapper is not None and hasattr(wrapper, "conv"):
        wrapper.conv = new
        if hasattr(wrapper, "c1"):
            wrapper.c1 = in_ch
    else:
        raise RuntimeError("Unsupported model structure for first-conv replacement")


def _patch_detection_trainer_get_model(in_ch: int = 6) -> None:
    # Patch the actual trainer model instance created inside Ultralytics.
    try:
        from ultralytics.models.yolo.detect.train import DetectionTrainer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import DetectionTrainer: {e}")

    if getattr(DetectionTrainer, "_sixch_patched", False):
        return

    orig_get_model = DetectionTrainer.get_model

    def get_model(self, cfg=None, weights=None, verbose=True):  # type: ignore
        m = orig_get_model(self, cfg=cfg, weights=weights, verbose=verbose)
        _expand_first_conv_model(m, in_ch=in_ch)
        try:
            first = m.model[0].conv
            print(f"[6CH] trainer model first conv in_channels={int(first.in_channels)}")
        except Exception:
            pass
        return m

    DetectionTrainer.get_model = get_model  # type: ignore
    DetectionTrainer._sixch_patched = True  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--model", type=str, default="yolo11n.pt")
    ap.add_argument("--imgsz", type=str, default="640")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--optimizer", type=str, default="SGD")
    ap.add_argument("--close_mosaic", type=int, default=5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs_paperlike")
    ap.add_argument("--name", type=str, default="exp_6ch")
    ap.add_argument("--motion_dirname", type=str, default="motion_images")
    args, unknown = ap.parse_known_args()
    _apply_kv_overrides(args, unknown)

    if not args.data:
        raise SystemExit("Missing --data")

    _patch_yolodataset_for_motion(motion_dirname=args.motion_dirname)
    _patch_detection_trainer_get_model(in_ch=6)

    from ultralytics import YOLO

    yolo = YOLO(args.model)
    # Note: Ultralytics trainer builds its own model instance. We patch DetectionTrainer.get_model
    # to ensure the model used for training is converted to 6ch.

    imgsz = _parse_imgsz(args.imgsz)

    yolo.train(
        data=args.data,
        imgsz=imgsz,
        epochs=int(args.epochs),
        batch=int(args.batch),
        optimizer=str(args.optimizer),
        close_mosaic=int(args.close_mosaic),
        device=str(args.device),
        project=str(args.project),
        name=str(args.name),
        pretrained=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
