from ultralytics import YOLO

# 1. YOLOv8 세그멘테이션 모델 불러오기
model = YOLO("yolov8l-seg.pt")  # YOLOv8 세그멘테이션 모델 (l: large, seg: 세그멘테이션)

# 2. 학습 데이터 및 설정
data_yaml = "data_30.yaml"  # 데이터셋 정의 YAML 파일 경로
epochs = 300                      # 학습 에폭 수
img_size = 640                   # 입력 이미지 크기clear
batch_size = 16                  # 배치 크기

# 3. 학습 시작 (Adam Optimizer 사용)
model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=1,  # GPU 번호 (1번 GPU 사용)
    project="../Yolo_saver",  # 결과 저장 경로
    name="data_30",
    optimizer="SGD",  # SGD Optimizer 설정
    patience=0
)
