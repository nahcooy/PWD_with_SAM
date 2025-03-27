import os
import argparse
import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM2UNet import SAM2UNet
from sklearn.metrics import jaccard_score, precision_recall_fscore_support, mean_absolute_error, confusion_matrix
from datetime import datetime

# Argument Parsing
parser = argparse.ArgumentParser("SAM2-UNet Training")
parser.add_argument("--hiera_path", type=str, required=True, help="Path to the pretrained SAM2 Hiera model")
parser.add_argument("--train_image_path", type=str, required=True, help="Path to training images")
parser.add_argument("--train_mask_path", type=str, required=True, help="Path to training masks")
parser.add_argument("--val_image_path", type=str, required=True, help="Path to validation images")
parser.add_argument("--val_mask_path", type=str, required=True, help="Path to validation masks")
parser.add_argument('--save_path', type=str, required=True, help="Path to store checkpoints")
parser.add_argument('--log_path', type=str, required=True, help="Path to save training logs")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to load checkpoint (optional)")
parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch for resuming training")
parser.add_argument("--epoch", type=int, default=100, help="Total number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
args = parser.parse_args()

# Logging function
def log_message(message, log_path):
    timestamp = datetime.now().strftime("%Y-%m-d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(formatted_message + '\n')

# Structure loss function
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# 클래스별 IoU 계산 함수
def calculate_class_iou(gt, pred, class_label):
    """ 특정 클래스 (0=배경, 1=객체)에 대한 IoU 계산 """
    gt_class = (gt == class_label).astype(int)
    pred_class = (pred == class_label).astype(int)

    if len(np.unique(gt_class)) < 2 or len(np.unique(pred_class)) < 2:
        return 0

    tn, fp, fn, tp = confusion_matrix(gt_class.flatten(), pred_class.flatten(), labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return iou

# Calculate evaluation metrics
def calculate_metrics(pred, target):
    pred = (pred > 0.5).int().to(target.device)
    target = target.int()

    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()

    iou_0 = calculate_class_iou(target_flat, pred_flat, class_label=0)  # Background IoU
    iou_1 = calculate_class_iou(target_flat, pred_flat, class_label=1)  # Object IoU
    miou = (iou_0 + iou_1) / 2  # Mean IoU

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='binary', zero_division=0)

    mAP = precision
    mae = mean_absolute_error(target_flat, pred_flat)

    return iou_0, iou_1, miou, f1, mAP, mae

# Validation function
def validate(model, dataloader, device, log_path):
    log_message("Starting validation", log_path)
    model.eval()
    total_loss, total_miou, total_f1, total_map, total_mae = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            pred0, pred1, pred2 = model(x)
            loss = structure_loss(pred2, target)

            _, _, miou, f1, mAP, mae = calculate_metrics(pred2, target)
            total_loss += loss.item()
            total_miou += miou
            total_f1 += f1
            total_map += mAP
            total_mae += mae

    n = len(dataloader)
    log_message(f"Validation - Loss: {total_loss / n:.4f}, mIoU: {total_miou / n:.4f}, "
                f"F1: {total_f1 / n:.4f}, mAP: {total_map / n:.4f}, MAE: {total_mae / n:.4f}", log_path)
    
    return total_loss / n, total_miou / n

# Main function
def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    log_file = os.path.join(args.log_path, "training_log_deep_prompt.txt")

    best_iou, best_loss = 0.0, float('inf')

    # 데이터셋 로드
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24)

    device = torch.device("cuda:0" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SAM2UNet(args.hiera_path).to(device)

    optim = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)

    for epoch in range(args.start_epoch, args.epoch):
        log_message(f"Starting epoch {epoch + 1}/{args.epoch}", log_file)
        model.train()
        for i, batch in enumerate(train_loader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)

            optim.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss = structure_loss(pred0, target) + structure_loss(pred1, target) + structure_loss(pred2, target)
            loss.backward()
            optim.step()
            scheduler.step()
            if i % 20 == 0:
                log_message(f"Epoch [{epoch + 1}/{args.epoch}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}", log_file)

        val_loss, val_iou = validate(model, val_loader, device, log_file)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-best-iou.pth'))
            log_message(f"[Checkpoint Saved] Best IoU model at epoch {epoch+1}", log_file)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-best-loss.pth'))
            log_message(f"[Checkpoint Saved] Best Loss model at epoch {epoch+1}", log_file)

        torch.save(model.state_dict(), os.path.join(args.save_path, 'SAM2-UNet-last.pth'))
        log_message("[Checkpoint Saved] Last checkpoint", log_file)
        
if __name__ == "__main__":
    main(args)