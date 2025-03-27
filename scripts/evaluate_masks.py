import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, mean_absolute_error

def calculate_class_iou(gt, pred, class_label):
    gt_class = (gt == class_label).astype(int)
    pred_class = (pred == class_label).astype(int)

    if len(np.unique(gt_class)) < 2 or len(np.unique(pred_class)) < 2:
        return 0

    tn, fp, fn, tp = confusion_matrix(gt_class.flatten(), pred_class.flatten(), labels=[0, 1]).ravel()
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

def calculate_metrics(pred, target):
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    iou_0 = calculate_class_iou(target_flat, pred_flat, class_label=0)
    iou_1 = calculate_class_iou(target_flat, pred_flat, class_label=1)
    miou = (iou_0 + iou_1) / 2

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='binary', zero_division=0)

    mAP = precision
    mae = mean_absolute_error(target_flat, pred_flat)

    return iou_0, iou_1, miou, f1, mAP, mae

def evaluate_directory(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    total = len(pred_files)

    total_iou0 = total_iou1 = total_miou = total_f1 = total_map = total_mae = 0.0
    for fname in pred_files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        
        if not os.path.exists(gt_path):
            print(f"[Warning] GT not found for: {fname}")
            continue

        pred = Image.open(pred_path).convert("L")
        gt = Image.open(gt_path).convert("L")

        if pred.size != gt.size:
            pred = pred.resize(gt.size, resample=Image.NEAREST)

        pred = np.array(pred) > 127
        gt = np.array(gt) > 127

        metrics = calculate_metrics(pred.astype(np.uint8), gt.astype(np.uint8))

        total_iou0 += metrics[0]
        total_iou1 += metrics[1]
        total_miou += metrics[2]
        total_f1 += metrics[3]
        total_map += metrics[4]
        total_mae += metrics[5]

    n = total
    print(f"📊 Evaluation Results on {n} samples:")
    print(f"Background IoU: {total_iou0 / n:.4f}")
    print(f"Object IoU:     {total_iou1 / n:.4f}")
    print(f"Mean IoU:       {total_miou / n:.4f}")
    print(f"F1 Score:       {total_f1 / n:.4f}")
    print(f"mAP (Precision):{total_map / n:.4f}")
    print(f"MAE:            {total_mae / n:.4f}")

# Example usage
if __name__ == "__main__":
    pred_mask_dir = "../predicts/SAM2-UNet/pred_50"      # 예측 결과 경로
    gt_mask_dir = "../data/val_50/gt"                    # GT 마스크 경로
    evaluate_directory(pred_mask_dir, gt_mask_dir)
