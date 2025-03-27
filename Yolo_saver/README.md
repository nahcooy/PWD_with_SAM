# 🗂️ Yolo_saver

이 디렉토리는 **YOLOv8 세그멘테이션 실험 결과를 저장**하는 공간입니다.

---

## 📁 구성 예시

```bash
Yolo_saver/
├── data_1/              # 1% 학습 결과
│   ├── weights/         # 학습된 모델 (.pt)
│   └── results.png      # 학습 로그 시각화

├── data_10/
├── data_30/
├── data_50/
├── data_100/

├── pred_masks_1/        # 1% 모델의 추론 마스크 저장 경로
├── pred_masks_10/
├── pred_masks_30/
├── pred_masks_50/
├── pred_masks_100/
```

---

## 📌 사용 목적

- 학습된 YOLOv8 모델의 `.pt` 파일을 저장
- 각 실험 비율에 따른 결과를 정리
- 추론 마스크(`.png`)도 실험 비율별로 별도 저장

---

## 📎 참고

- 학습 결과는 `train_Yolov8_*.py` 에서 `project="../Yolo_saver"` 옵션을 통해 자동 저장됩니다.
- 추론 결과는 `predict_Yolov8_*.py` 에서 `output_dir` 경로로 저장됩니다.