# 🌲 Pine Wilt Disease (PWD) Segmentation with YOLOv8 & SAM2-UNet

이 프로젝트는 YOLOv8 및 SAM2-UNet을 활용한 소나무재선충병(Pine Wilt Disease, PWD) 세그멘테이션을 위한 실험 및 분석을 수행합니다.

---

## 📂 프로젝트 구조
```bash
PWD_with_SAM/
├── data/                 # 원본 및 샘플링 데이터
├── SAM2-UNet_pwd/        # SAM2-UNet 모델 학습 및 예측
├── SAM2-UNet_saver/      # SAM2-UNet 학습 결과 (체크포인트)
├── Yolo_pwd/             # YOLOv8 모델 학습 및 예측
├── Yolo_saver/           # YOLOv8 학습 결과 (체크포인트)
├── scripts/              # 데이터 전처리 및 평가 스크립트
└── predicts/             # 모델 예측 마스크 저장 경로
```
---

## 🚀 시작하기

1️⃣ 데이터 준비 (data/)

- 원본 데이터를 준비하고, 데이터 하위 샘플링 실행:
  
python scripts/split_data.py

자세히 보기 👉 [data/README.md](data/README.md)

2️⃣ 모델 학습 & 예측

- YOLOv8
  
cd Yolo_pwd
pip install -r requirement.txt
python train_Yolov8_10.py
python predict_Yolov8_10.py

자세히 보기 👉 [Yolo_pwd/README.md](Yolo_pwd/README.md)

- SAM2-UNet
```bash
cd SAM2-UNet_pwd
pip install -r requirements.txt

python new_train.py \
  --hiera_path "./sam2_hiera_large.pt" \
  --train_image_path "../data/train_10/images" \
  --train_mask_path "../data/train_10/gt" \
  --val_image_path "../data/val_10/images" \
  --val_mask_path "../data/val_10/gt" \
  --save_path "../SAM2-UNet_saver/fp_10" \
  --log_path "../SAM2-UNet_saver/fp_10" \
  --epoch 200 > sam2unet_10.txt

python pred_mask_SAM2-UNet.py \
  --hiera_path "./sam2_hiera_large.pt" \
  --val_image_path "../data/val_10/images" \
  --val_mask_path "../data/val_10/gt" \
  --checkpoint_path "../SAM2-UNet_saver/fp_10/SAM2-UNet-best-iou.pth" \
  --result_dir "../predicts/SAM2-UNet/pred_10"
```
자세히 보기 👉 [SAM2-UNet_pwd/README.md](SAM2-UNet_pwd/README.md)

---

## 📁 세부 디렉토리 설명

자세히 보기 👉 각 디렉토리의 개별 README.md 파일 참조

---

## 🧪 성능 평가

모델 예측 마스크의 성능 평가를 원하면 다음을 실행합니다:

python scripts/evaluate_masks.py

---

## 📋 필수 환경
```bash

PyTorch 1.9+
ultralytics(YOLO) latest
numpy latest
Pillow latest
scikit-learn latest
```
각 디렉토리 내 requirements.txt 파일 참조하여 설치합니다.

---

📬 문의사항이 있으면 언제든 연락주세요.  
🚩 프로젝트 소유자: nahcooy  
🚩 버전: 2024.03
