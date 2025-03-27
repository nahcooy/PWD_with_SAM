# 🧠 SAM2-UNet_saver

이 디렉토리는 **SAM2-UNet 세그멘테이션 실험 결과를 저장**하는 공간입니다.

---

## 📁 구성 예시

```bash
SAM2-UNet_saver/
├── fp_1/                    # 1% 학습 결과
│   ├── SAM2-UNet-best-iou.pth     # 가장 높은 IoU 모델
│   ├── SAM2-UNet-best-loss.pth    # 가장 낮은 Loss 모델
│   ├── SAM2-UNet-last.pth         # 마지막 에폭 저장 모델
│   └── training_log_deep_prompt.txt  # 학습 로그 파일

├── fp_10/
├── fp_30/
├── fp_50/
├── fp/                      # 전체 데이터(100%) 학습 결과
```

---

## 📌 사용 목적

- 학습된 SAM2-UNet 모델의 체크포인트(.pth) 파일 저장
- 각 실험 비율에 따른 학습 로그(`training_log_deep_prompt.txt`) 저장
- 추론 시 사용할 best 모델 접근을 용이하게 관리

---

## 📎 참고

- 학습 결과는 `new_train.py` 실행 시 `--save_path` 및 `--log_path` 인자로 지정된 경로에 저장됩니다.
- 추론 시 해당 디렉토리 내의 `SAM2-UNet-best-iou.pth` 파일을 사용
