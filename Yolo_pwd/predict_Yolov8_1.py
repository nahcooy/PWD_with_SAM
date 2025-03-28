from ultralytics import YOLO
import os
import cv2

# 1. 모델 로드 (학습된 best.pt 경로)
model = YOLO("../Yolo_saver/data_10/weights/best.pt")

# 2. 예측할 이미지 디렉토리 경로 설정
image_dir = "../data/val_1/images"
output_dir = "../predicts/yolo/pred_1"
os.makedirs(output_dir, exist_ok=True)

# 3. 이미지 순회하며 예측 수행
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
for image_name in sorted(image_files):
    image_path = os.path.join(image_dir, image_name)
    results = model(image_path, save=False, imgsz=640)

    # 4. segmentation mask 추출
    for i, r in enumerate(results):
        if r.masks is not None:
            mask = r.masks.data[0].cpu().numpy()  # 첫 번째 mask만 저장
            mask = (mask * 255).astype("uint8")
            save_path = os.path.join(output_dir, image_name)
            cv2.imwrite(save_path, mask)
        else:
            print(f"[경고] {image_name}에서 마스크를 찾을 수 없습니다.")

            # 원본 이미지 크기 가져오기
            img = cv2.imread(image_path)
            h, w = img.shape[:2]

            # 검정 마스크 생성 및 저장
            black_mask = np.zeros((h, w), dtype="uint8")
            cv2.imwrite(save_path, black_mask)
