# Project summary
This repository contains a two-stage pipeline: (1) a YOLOv11n object detector trained to detect faces and predict gender (male / female), and (2) an EfficientNet_V2_S–based classifier fine-tuned to recognize 16 celebrity identities from cropped face images. The detector produces face crops which the classifier then labels. See the full report for details and results.

---

## Table of contents
- [Overview](#overview)  
- [Dataset](#dataset)  
- [YOLO detector (gender)](#yolo-detector-gender)  
- [Classifier (celebrity recognition)](#classifier-celebrity-recognition)  
- [Pipeline & evaluation](#pipeline--evaluation)  
- [How to reproduce / run](#how-to-reproduce--run)  
- [Files in this repo](#files-in-this-repo)  
- [Placeholders for images (add yours)](#placeholders-for-images-add-yours)  
- [License & contact](#license--contact)

---

## Overview
This work integrates two models to form a lightweight, accurate pipeline for gender detection and identity recognition from images:
- **YOLOv11n**: lightweight object detector for face detection + gender class (male/female).  
- **EfficientNet_V2_S**: transfer-learned classifier fine-tuned to recognize 16 celebrity identities on cropped face images.  

Both models were trained with augmentation and best-practice regularization (MixUp for the classifier).

---

## Dataset
- **Combined sources**:
  - *Dataset A*: ~100 manually annotated images (bounding boxes + gender).
  - *Dataset B*: Roboflow import (pre-annotated).  
- **Annotation format**: YOLO-style `.txt` per image:  
  `<class_id> <x_center> <y_center> <width> <height>` (normalized).  
- **Classes**:
  - `0` → Male  
  - `1` → Female  
- **Dataset split / sizes**:
  - Train: 1872 images  
  - Validate: 225 images  

---

## YOLO detector (gender)
**Model**: YOLOv11n (nano) — chosen for fast inference and small model size.  

**Training details**
- Framework: PyTorch (Ultralytics YOLOv11).  
- Training environment: Kaggle Notebook (Tesla T4).  
- Epochs: 100  
- Image size: 640×640 (resized with padding)  
- Batch size: 16  

**Performance (summary)**  
- Precision: 85.7%  
- Recall: 68.9%  
- mAP@0.5: 77.8%  
- mAP@0.5:0.95: 51.8%  

---

## Classifier (celebrity recognition)
**Backbone**: `EfficientNet_V2_S` pre-trained on ImageNet1K (IMAGENET1K_V1 weights).  

**Head (custom)**
- Dropout(0.5)  
- Linear(1280 → 512) → ReLU  
- Dropout(0.3)  
- Linear(512 → 16) (final logits for 16 classes)  

**Data & preprocessing**
- Source: Kaggle competition dataset; input files loaded from `.npy` (`faces_cropped.npy`, `labels_cropped.npy`).  
- Input size: 224×224 RGB (already face-cropped).  
- Unknown labels (`-1`) remapped to class `15` to keep labels in `0..15`.  

**Augmentations (Albumentations)**
- HorizontalFlip (p=0.5)  
- Rotate (±30°)  
- Hue/Saturation/Value jitter  
- RandomBrightnessContrast  
- CoarseDropout (1–2 holes covering 10–20% area)  
- Normalization + ToTensor  

**Training strategy**
- **Phase 1**: Train classifier head only (backbone frozen).  
  - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4  
  - Epochs: 100  
  - Loss: CrossEntropy with MixUp (alpha=0.4)  
- **Phase 2**: Unfreeze last 5 blocks + head; fine-tune.  
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-5  
  - Epochs: 50  

**Result**
- Validation accuracy reported as **>99%** on the hold-out set after fine-tuning.  

---

## Pipeline & evaluation
1. Run YOLOv11n detector on input images / video frames → obtain bounding boxes + gender.  
2. Crop detected faces (use bounding box coordinates, optionally pad) and resize to 224×224.  
3. Feed crops into EfficientNet classifier → get celebrity identity probabilities.  
4. Combine outputs for final visualization (bounding box + gender label + identity label + confidence).  

**Notes on evaluation**
- YOLO metrics reported above (Precision/Recall/mAP).  
- Classifier evaluated separately on cropped-face validation set.  
- The classifier was used to label YOLO crops during evaluation/visualization to validate end-to-end pipeline performance.  

---

## How to reproduce / run
> These are example commands—adjust paths and hyperparameters to match your environment.

1. **Prepare datasets**
```bash
# Place images and YOLO .txt labels in `data/images/` and `data/labels/`
# Ensure train/val splits are set, or create YAML pointing to the splits.
````

2. **Train YOLOv11n**

```bash
# Using an Ultralytics-style interface (example)
yolo train model=yolov11n.pt data=data/your_yolov11_dataset.yaml imgsz=640 batch=16 epochs=100
```

3. **Train classifier**

```bash
# Example PyTorch-like training (pseudocode)
python train_classifier.py --data faces_cropped.npy --labels labels_cropped.npy \
                           --backbone efficientnet_v2_s --img-size 224 --batch-size 32 \
                           --phase1-epochs 100 --phase2-epochs 50 --mixup-alpha 0.4
```

4. **Run pipeline (inference)**

```bash
python run_pipeline.py --yolo-weights runs/yolo/exp/weights/best.pt \
                       --clf-weights runs/clf/exp/weights/best.pth \
                       --source path/to/images_or_video
```

---

## Files in this repo

```
README.md                # This file
data/
  images/                # raw images
  labels/                # YOLO .txt label files
  faces_cropped.npy      # cropped-face array (optional)
  labels_cropped.npy     # labels for classifier (optional)
models/
  yolov11n/              # YOLO training outputs / weights
  efficientnet_v2_s/     # classifier weights
notebooks/
  yolo_pipeline.ipynb    # training / debugging notebook
  classifier.ipynb       # training / debugging notebook
scripts/
  train_yolo.sh
  train_classifier.sh
  run_pipeline.py
report.pdf               # detailed report and full results
images/                  # add images/figures here (placeholders listed below)
```

---

## Placeholders for images (add yours into `images/` folder)

* YOLO training curves / PR curve:
  `![YOLO training curves](images/yolo_training_curves.png)`

* YOLO sample detections (visualization of bounding boxes + gender):
  `![YOLO sample detections](images/yolo_detections.png)`

* Classifier training curves / confusion matrix:
  `![Classifier training curves](images/classifier_training.png)`

* Example pipeline output (bbox + gender + identity labels):
  `![Pipeline example](images/pipeline_example.png)`

---

## Notes, caveats & suggestions

* The YOLO detector was trained on a relatively small custom dataset (~2k images). Consider collecting/annotating more images (diverse lighting, poses, occlusion) to improve recall.
* The classifier’s very high validation accuracy suggests strong fit — verify performance on a held-out real-world test set to ensure robustness to deployment domain shift.
* If running realtime on edge devices, YOLOv11n is a good lightweight choice; consider further pruning / quantization for embedded deployment.

---

## License & contact

* **License:** This project is licensed under the MIT License.
* **Author / contact:** Eman Murtaza Turk — https://github.com/turkEman

```
