#!/bin/bash
# Run RGB + TNF training for folds 1,2,3,5
# Fold 4 already done

set -e
PROJECT=/workspace/proj1
cd $PROJECT

for FOLD in 1 2 3 5; do
    echo "========================================"
    echo "FOLD $FOLD - RGB baseline (no HSV)"
    echo "========================================"
    # Prepare YOLO data if needed
    if [ ! -f "$PROJECT/Data/yolo_fold${FOLD}/data.yaml" ] || [ -z "$(ls $PROJECT/Data/yolo_fold${FOLD}/images/train/ 2>/dev/null)" ]; then
        python Data/convert_fold_to_yolo.py --fold $FOLD --out Data/yolo_fold${FOLD} --mode copy
    fi
    # Train RGB baseline
    python -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.train(
    data='$PROJECT/Data/yolo_fold${FOLD}/data.yaml',
    epochs=300, imgsz=640, batch=32, device=0,
    project='$PROJECT/runs/5fold_cv', name='rgb_no_hsv_fold${FOLD}',
    exist_ok=True, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, verbose=False
)
"

    echo "========================================"
    echo "FOLD $FOLD - RAFT+Attn+TNF"
    echo "========================================"
    # Generate RAFT+TNF data if needed
    if [ ! -f "$PROJECT/Data/yolo_fold${FOLD}_raft_attn_tnf/data.yaml" ] || [ -z "$(ls $PROJECT/Data/yolo_fold${FOLD}_raft_attn_tnf/images/train/ 2>/dev/null)" ]; then
        python Data/gen_raft_flow_attn.py \
            --in_root  Data/yolo_fold${FOLD} \
            --out_root Data/yolo_fold${FOLD}_raft_attn_tnf \
            --normalize_dt --attn_mode mag --attn_gamma 0.5 --attn_blur 3 --device cuda
    fi
    # Train TNF model (standard 3ch YOLO, same as fold4 raft_tnf_fold4_motionaug)
    python -c "
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
m.train(
    data='$PROJECT/Data/yolo_fold${FOLD}_raft_attn_tnf/data.yaml',
    epochs=300, imgsz=640, batch=32, device=0,
    project='$PROJECT/runs/5fold_cv', name='raft_tnf_fold${FOLD}_motionaug',
    exist_ok=True, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0, verbose=False
)
"

    echo "FOLD $FOLD DONE"
done

echo "========================================"
echo "ALL FOLDS COMPLETE - collecting results"
echo "========================================"
python -c "
from ultralytics import YOLO
import numpy as np, os

base = '/workspace/proj1/runs/5fold_cv'
folds = [1,2,3,4,5]
rgb_results, tnf_results = [], []

for fold in folds:
    for name, yaml_path, store in [
        (f'rgb_no_hsv_fold{fold}',        f'/workspace/proj1/Data/yolo_fold{fold}/data.yaml',              rgb_results),
        (f'raft_tnf_fold{fold}_motionaug', f'/workspace/proj1/Data/yolo_fold{fold}_raft_attn_tnf/data.yaml', tnf_results),
    ]:
        pt = f'{base}/{name}/weights/best.pt'
        if not os.path.exists(pt):
            print(f'MISSING: {pt}'); continue
        m = YOLO(pt).val(data=yaml_path, imgsz=640, device=0, verbose=False)
        r = m.results_dict
        res = dict(fold=fold,
                   P=r['metrics/precision(B)'], R=r['metrics/recall(B)'],
                   mAP50=r['metrics/mAP50(B)'], mAP5095=r['metrics/mAP50-95(B)'])
        store.append(res)
        print(f'{name}: P={res[\"P\"]:.4f} R={res[\"R\"]:.4f} mAP50={res[\"mAP50\"]:.4f} mAP50-95={res[\"mAP5095\"]:.4f}')

for label, results in [('RGB', rgb_results), ('TNF', tnf_results)]:
    if not results: continue
    for k in ['P','R','mAP50','mAP5095']:
        vals = [r[k] for r in results]
        print(f'{label} {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
"
