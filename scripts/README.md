# 🛠️ scripts - Dataset & Evaluation Tools

이 디렉토리는 PWD 세그멘테이션 실험을 위한 
데이터 전처리 및 결과 평가 스크립트를 포함합니다.

각 스크립트는 실험 준비와 성능 분석에 핵심적으로 활용됩니다.

------------------------------------------
📁 디렉토리 구성
------------------------------------------
1. split_data.py
    - train/ 및 val/ 데이터를 기반으로 1%, 10%, 30%, 50% 비율로 샘플링
    - 결과는 data/train_1, data/val_1, ..., data/train_50 등의 폴더에 저장
    - 실행 전 `data/train/`, `data/val/` 폴더가 미리 존재해야 함

    ✅ 실행 예시:
        python split_data.py

2. evaluate_masks.py
    - 예측된 마스크(.png)와 GT 마스크를 비교하여 성능 지표 계산
    - 평가 결과로 `Background IoU`, `Object IoU`, `Mean IoU`, `F1 Score`, `mAP`, `MAE` 등을 출력

    ✅ 실행 예시:
        python evaluate_masks.py

------------------------------------------
📦 의존성 설치
------------------------------------------

필수 패키지 설치는 아래 명령어로 가능합니다:

    pip install -r requirements.txt
