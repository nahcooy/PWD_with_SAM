import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2  # OpenCV 사용
from dataset import FullDataset
from torch.utils.data import DataLoader
from SAM2UNet import SAM2UNet
import random

# Seed 설정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def apply_overlay(image, pred_mask, gt_mask, alpha=0.3):
    """원본 이미지 위에 GT 마스크 테두리와 예측 마스크 오버레이 적용"""
    overlay = Image.new('RGBA', image.size, (255, 0, 0, 0))  # 빨간색 오버레이
    draw = ImageDraw.Draw(overlay)

    # GT 마스크: 파란색 테두리 그리기
    gt_mask_np = gt_mask.squeeze().cpu().numpy().astype('uint8')
    gt_resized = cv2.resize(gt_mask_np, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(gt_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        polygon = [(int(point[0][0]), int(point[0][1])) for point in contour]
        draw.line(polygon + [polygon[0]], fill=(0, 0, 255, 255), width=3)  # 파란색 테두리

    # 예측 마스크: 빨간색 오버레이 적용
    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype('uint8')
    pred_resized = cv2.resize(pred_mask_np, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    pred_img = Image.fromarray((pred_resized * 255).astype('uint8')).convert('L')
    overlay.paste((255, 0, 0, int(255 * alpha)), mask=pred_img)  # 빨간색 오버레이

    # 오버레이와 원본 이미지 합성
    image = image.convert('RGBA')
    return Image.alpha_composite(image, overlay)

def validate_and_overlay(model, dataloader, device, output_dir, max_samples=20):
    """모델 검증 및 오버레이 저장, 최대 max_samples개의 샘플에 대해 실행"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0  # 처리한 샘플 수를 추적

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break  # max_samples개의 샘플을 처리하면 종료

            x = batch['image'].to(device)  # 입력 이미지
            gt_mask = batch['label'].to(device)  # GT 마스크

            # 모델 예측 수행
            _, _, pred_mask = model(x)
            pred_mask = (torch.sigmoid(pred_mask) > 0.5).int()  # 예측 마스크 생성

            # 결과 오버레이 적용 및 저장
            for j in range(x.size(0)):
                if sample_count >= max_samples:
                    break  # max_samples개의 샘플을 처리하면 종료

                # 입력 이미지 변환 및 저장
                input_image = x[j].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)로 변환
                input_image = (input_image * 255).clip(0, 255).astype('uint8')
                original_image = Image.fromarray(input_image)

                # 오버레이 적용
                overlay = apply_overlay(original_image, pred_mask[j], gt_mask[j])

                # 오버레이 이미지 저장
                overlay_save_path = os.path.join(output_dir, f"overlay_{i}_{j}.png")
                overlay.save(overlay_save_path)
                print(f"Saved overlay: {overlay_save_path}")

                # 예측 마스크 저장 (.npy 파일)
                pred_mask_np = pred_mask[j].cpu().numpy()
                pred_mask_save_path = os.path.join(output_dir, f"pred_mask_{i}_{j}.npy")
                np.save(pred_mask_save_path, pred_mask_np)
                print(f"Saved pred mask (npy): {pred_mask_save_path}")

                # 예측 마스크 저장 (.png 파일)
                # 필요에 따라 차원 축소
                if pred_mask_np.ndim > 2:
                    pred_mask_np = np.squeeze(pred_mask_np)

                pred_mask_img = Image.fromarray((pred_mask_np * 255).astype('uint8'), mode='L')
                pred_mask_img_save_path = os.path.join(output_dir, f"pred_mask_{i}_{j}.png")
                pred_mask_img.save(pred_mask_img_save_path)
                print(f"Saved pred mask (png): {pred_mask_img_save_path}")

                sample_count += 1  # 처리된 샘플 수 증가

                if sample_count >= max_samples:
                    break  # max_samples가 처리되면 루프 종료

def load_checkpoint(checkpoint_path, model, device):
    """체크포인트에서 모델 가중치만 로드"""
    print(f"Loading model parameters from {checkpoint_path}")
    
    # 체크포인트에서 모델 가중치만 불러오기
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 모델을 GPU로 이동 (필요한 경우)
    model.to(device)
    
    print(f"Model parameters loaded from {checkpoint_path}")
    return model

def main(args):
    # 데이터셋 로드
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화 및 체크포인트 로드
    model = SAM2UNet(args.hiera_path)
    model = load_checkpoint(args.checkpoint_path, model, device)

    print("Start validation and overlay generation...")
    validate_and_overlay(model, val_dataloader, device, args.output_dir)
    print("Validation and overlay generation complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("SAM2-UNet Validation and Overlay")
    parser.add_argument("--hiera_path", type=str, required=True, help="path to the sam2 pretrained hiera")
    parser.add_argument("--val_image_path", type=str, required=True, help="path to the images for validation")
    parser.add_argument("--val_mask_path", type=str, required=True, help="path to the GT masks for validation")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="path to the saved model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save the overlay images")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_samples", default=20, type=int, help="number of samples to process during validation")

    args = parser.parse_args()
    main(args)
