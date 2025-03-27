import os
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import FullDataset
from SAM2UNet import SAM2UNet
import numpy as np
from PIL import Image

# 마스크 저장 함수
def save_mask(mask, output_path, image_path):
    os.makedirs(output_path, exist_ok=True)
    image_name = os.path.basename(image_path).split('.')[0]  # 경로에서 파일명만 추출
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # 흰색(255)와 검정색(0)으로 변환
    mask_save_path = os.path.join(output_path, f"{image_name}.png")
    mask_image.save(mask_save_path)

# 모델 평가 함수
def generate_masks(model, dataloader, device, result_dir):
    model.eval()
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"Processing {i + 1}/{len(dataloader)} batches")

            # 입력 이미지 가져오기
            x = batch['image'].to(device)
            image_path = dataloader.dataset.images[i]  # 데이터셋의 이미지 경로 가져오기

            # 모델을 통해 예측된 마스크 생성
            pred0, _, _ = model(x)  # 첫 번째 출력 사용
            pred_mask = torch.sigmoid(pred0).squeeze().cpu().numpy() > 0.5  # Threshold 0.5 적용
            save_mask(pred_mask, result_dir, image_path)

# 메인 함수
def main(args):
    # Validation dataset 로드
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 352, mode='val')

    # DataLoader 생성 (batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = SAM2UNet(args.hiera_path)

    # checkpoint에서 모델 가중치만 불러오기
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)  # 수정된 부분
    model.to(device)

    # 모델 예측 마스크 생성 및 저장
    print("Start generating masks")
    generate_masks(model, val_dataloader, device, args.result_dir)
    print("Mask generation is done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SAM2-UNet Mask Generation")
    parser.add_argument("--hiera_path", type=str, required=True, help="path to the pretrained SAM2 model")
    parser.add_argument("--val_image_path", type=str, required=True, help="path to the validation images")
    parser.add_argument("--val_mask_path", type=str, required=True, help="path to the validation masks")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="path to the model checkpoint (.pth)")
    parser.add_argument("--batch_size", default=1, type=int)  # batch_size=1로 설정
    parser.add_argument("--result_dir", type=str, required=True, help="directory to save generated masks")

    args = parser.parse_args()
    main(args)
