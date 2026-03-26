# HAM10000 Skin Lesion Classification

> A deep learning project focusing on transfer learning and fine-tuning strategies for medical image classification.

---

## 📌 Overview (English)

This project focuses on skin lesion classification using the HAM10000 dataset.
It applies transfer learning with deep learning models such as ResNet50 and EfficientNet-B2.

Multiple fine-tuning strategies are explored and compared:

* Head-only training
* Full fine-tuning
* Layer-wise fine-tuning

---

## 📌 개요 (Korean)

본 프로젝트는 HAM10000 데이터셋을 활용한 피부 병변 분류 문제를 다룹니다.
ResNet50과 EfficientNet-B2 모델을 활용한 전이학습(Transfer Learning)을 적용하였습니다.

다음과 같은 다양한 학습 전략을 비교 분석합니다:

* Head-only 학습
* 전체 fine-tuning
* 일부 layer fine-tuning

---

## 🧠 Dataset

* HAM10000 (Human Against Machine with 10000 training images)
* 7-class skin lesion classification

### Classes

* akiec: Actinic keratoses / Bowen's disease
* bcc: Basal cell carcinoma
* bkl: Benign keratosis-like lesions
* df: Dermatofibroma
* mel: Melanoma
* nv: Melanocytic nevi
* vasc: Vascular lesions

---

## 🚀 Models

* ResNet50
* EfficientNet-B2

---

## ⚙️ Training Strategies

### 1. Head-only Training

* Backbone freeze
* Only classifier head is trained

### 2. Full Fine-tuning

* Entire network is trained

### 3. Layer-wise Fine-tuning

* Only specific layers (e.g., layer4) are trained

---

## 🧪 Experiments

* ResNet50 head-only training
* ResNet50 full fine-tuning
* ResNet50 layer-wise fine-tuning
* EfficientNet-B2 full fine-tuning

---

## 📊 Evaluation Metrics

* Accuracy
* Macro F1-score
* Validation Loss

Macro F1-score is especially important due to class imbalance in HAM10000.

---

## 📂 Project Structure

```text
ham10000/
├── ham10000_finetune.py
├── HAM10000/
│   └── README.txt
├── runs_ham10000/
│   └── README.txt
├── README.md
├── .gitignore
```

---

## ⚙️ How to Run

```bash
python ham10000_finetune.py \
  --data_dir "./HAM10000" \
  --model resnet50 \
  --epochs_head 3 \
  --epochs_ft 8 \
  --batch_size 32 \
  --amp
```

### Available model options

* resnet50
* efficientnet_b2

---

## 📦 Output

Trained model checkpoints are saved in:

```text
runs_ham10000/
```

Examples:

* best_head.pt
* best_finetuned.pt

---

## ⚠️ Dataset & Checkpoints

The dataset and trained model weights are **NOT included** in this repository due to size limitations.

Please manually download the HAM10000 dataset and place it in the `HAM10000/` directory.

Model checkpoints (e.g., best_head.pt, best_finetuned.pt) will be saved in `runs_ham10000/`.

---

## ⚠️ 데이터 및 모델 파일 안내

데이터셋과 학습된 모델 파일은 용량 문제로 저장소에 포함되어 있지 않습니다.

HAM10000 데이터셋을 직접 다운로드하여 `HAM10000/` 폴더에 넣어야 합니다.

모델 체크포인트는 `runs_ham10000/` 폴더에 저장됩니다.

---

## 📉 Analysis

* Confusion matrix analysis
* Class imbalance impact
* Misclassified sample analysis

---

## 📌 Future Work

* Apply more advanced architectures (e.g., ConvNeXt)
* Improve data augmentation strategies
* Handle class imbalance with sampling techniques
* Add training/validation visualization
* Hyperparameter tuning

---

## 📄 License

MIT License
