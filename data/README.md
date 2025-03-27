# 📂 data/

이 폴더는 세그멘테이션 실험을 위한 **원본 데이터셋 및 샘플링된 하위 데이터셋**을 저장하는 공간입니다.

---

## 📁 기본 구조

```bash
data/
├── train/          # 학습용 원본 데이터셋
│   ├── images/     # 원본 이미지 (.png)
│   ├── labels/     # YOLO 형식 Polygon 라벨 (.txt)
│   └── gt/         # GT 마스크 이미지 (.png)
├── val/            # 검증용 원본 데이터셋
│   ├── images/     # 원본 이미지 (.png)
│   ├── labels/     # YOLO 형식 Polygon 라벨 (.txt)
│   └── gt/         # GT 마스크 이미지 (.png)
├── train_1/        # 1% 샘플링 학습 데이터
├── val_1/          # 1% 샘플링 검증 데이터
├── train_10/       # 10% 샘플링 학습 데이터
├── val_10/         # 10% 샘플링 검증 데이터
├── train_30/       # 30% 샘플링 학습 데이터
├── val_30/         # 30% 샘플링 검증 데이터
├── train_50/       # 50% 샘플링 학습 데이터
└── val_50/         # 50% 샘플링 검증 데이터
```

---

## ✅ 폴더별 역할

- `images/`: 원본 이미지 파일 (예: `0001.png`, `0002.png`, ...)
- `labels/`: YOLO 포맷의 `.txt` 파일 (Polygon 좌표 기반 GT 라벨)
- `gt/`: 해당 이미지의 정답 마스크 이미지 (`.png`, binary mask)

---

## 🛠 샘플링 데이터 생성 방법

`train_1`, `val_10`, `train_50` 등의 샘플 데이터는 아래 스크립트를 통해 자동 생성할 수 있습니다.

### ▶️ 실행 명령어:

```bash
python scripts/split_data.py
```

---

## ⚠️ 주의사항

`data/train/`, `data/val/` 폴더가 **미리 존재해야 하며**,  
각 폴더 안에 반드시 다음 세 폴더가 있어야 합니다:

- `images/`: 원본 이미지 (raw image)
- `labels/`: YOLO 형식의 GT polygon 텍스트 파일 (`.txt`)
- `gt/`: 각 이미지의 GT 마스크 이미지 (`.png`)

이 구조를 기반으로 `split_data.py`가 랜덤 샘플링을 수행하여 하위 데이터셋을 자동 생성합니다.
