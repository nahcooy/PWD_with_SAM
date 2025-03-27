# 🧠 SAM2-UNet_pwd - Segmentation Training


이 디렉토리는 SAM2-UNet 기반의 세그멘테이션 학습 및 추론 스크립트를 포함합니다.
다양한 데이터 비율을 대상으로 SAM2-UNet 학습 및 예측을 수행하며,
입력 이미지 및 출력 디렉토리는 실행 시 인자로 지정할 수 있습니다.

------------------------------------------
📁 디렉토리 구성
------------------------------------------
- new_train.py                : SAM2-UNet 학습용 메인 스크립트
- pred_mask_SAM2-UNet.py      : 학습된 모델을 통한 마스크 예측 스크립트
- SAM2UNet.py                 : 모델 구조 정의
- dataset.py                  : 커스텀 데이터셋 로더
- sam2/                       : SAM2 관련 백엔드 코드
- sam2_configs/               : 하이레벨 모델 설정 파일 (.yaml)
- requirements.txt            : 의존 패키지 목록

------------------------------------------
📦 의존성 설치
------------------------------------------
아래 명령어로 필수 패키지를 설치할 수 있습니다:

    pip install -r requirements.txt

------------------------------------------
🚀 학습 실행 예시
------------------------------------------
# 1%
```bash
python new_train.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --train_image_path "../data/train_1/images" \\
  --train_mask_path "../data/train_1/gt" \\
  --val_image_path "../data/val_1/images" \\
  --val_mask_path "../data/val_1/gt" \\
  --save_path "../SAM2-UNet_saver/fp_1" \\
  --log_path "../SAM2-UNet_saver/fp_1" \\
  --epoch 200 > sam2unet_1.txt
```
```bash
# 10%
python new_train.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --train_image_path "../data/train_10/images" \\
  --train_mask_path "../data/train_10/gt" \\
  --val_image_path "../data/val_10/images" \\
  --val_mask_path "../data/val_10/gt" \\
  --save_path "../SAM2-UNet_saver/fp_10" \\
  --log_path "../SAM2-UNet_saver/fp_10" \\
  --epoch 200 > sam2unet_10.txt
```
```bash
# 30%
python new_train.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --train_image_path "../data/train_30/images" \\
  --train_mask_path "../data/train_30/gt" \\
  --val_image_path "../data/val_30/images" \\
  --val_mask_path "../data/val_30/gt" \\
  --save_path "../SAM2-UNet_saver/fp_30" \\
  --log_path "../SAM2-UNet_saver/fp_30" \\
  --epoch 200 > sam2unet_30.txt
```
```bash
# 50%
python new_train.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --train_image_path "../data/train_50/images" \\
  --train_mask_path "../data/train_50/gt" \\
  --val_image_path "../data/val_50/images" \\
  --val_mask_path "../data/val_50/gt" \\
  --save_path "../SAM2-UNet_saver/fp_50" \\
  --log_path "../SAM2-UNet_saver/fp_50" \\
  --epoch 200 > sam2unet_50.txt
```
```bash
# 100%
python new_train.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --train_image_path "../data/train/images" \\
  --train_mask_path "../data/train/gt" \\
  --val_image_path "../data/val/images" \\
  --val_mask_path "../data/val/gt" \\
  --save_path "../SAM2-UNet_saver/fp" \\
  --log_path "../SAM2-UNet_saver/fp" \\
  --epoch 200 > sam2unet.txt
```
------------------------------------------
🧪 추론 실행 예시
------------------------------------------
# 1%
```bash
python pred_mask_SAM2-UNet.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --val_image_path "../data/val_1/images" \\
  --val_mask_path "../data/val_1/gt" \\
  --checkpoint_path "../SAM2-UNet_saver/fp_1/SAM2-UNet-best-iou.pth" \\
  --result_dir "../predicts/SAM2-UNet/pred_1"
```
```bash
# 10%
python pred_mask_SAM2-UNet.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --val_image_path "../data/val_10/images" \\
  --val_mask_path "../data/val_10/gt" \\
  --checkpoint_path "../SAM2-UNet_saver/fp_10/SAM2-UNet-best-iou.pth" \\
  --result_dir "../predicts/SAM2-UNet/pred_10"
```
```bash
# 30%
python pred_mask_SAM2-UNet.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --val_image_path "../data/val_30/images" \\
  --val_mask_path "../data/val_30/gt" \\
  --checkpoint_path "../SAM2-UNet_saver/fp_30/SAM2-UNet-best-iou.pth" \\
  --result_dir "../predicts/SAM2-UNet/pred_30"
```
```bash
# 50%
python pred_mask_SAM2-UNet.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --val_image_path "../data/val_50/images" \\
  --val_mask_path "../data/val_50/gt" \\
  --checkpoint_path "../SAM2-UNet_saver/fp_50/SAM2-UNet-best-iou.pth" \\
  --result_dir "../predicts/SAM2-UNet/pred_50"
```
```bash
# 100%
python pred_mask_SAM2-UNet.py \\
  --hiera_path "./sam2_hiera_large.pt" \\
  --val_image_path "../data/val/images" \\
  --val_mask_path "../data/val/gt" \\
  --checkpoint_path "../SAM2-UNet_saver/fp/SAM2-UNet-best-iou.pth" \\
  --result_dir "../predicts/SAM2-UNet/pred"
```

※ 추론 결과 마스크는 predicts/SAM2-UNet/ 디렉토리에 저장됩니다.
   각 실험 비율별로 pred_1, pred_10, pred_30, pred_50, pred_100 형태로 나뉩니다.
"""

Path("SAM2-UNet_pwd_README.txt").write_text(readme_text.strip(), encoding="utf-8")
"SAM2-UNet_pwd_README.txt 파일이 저장되었습니다."
