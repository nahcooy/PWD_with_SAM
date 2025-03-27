import os
import random
import shutil

def sample_and_save(src_dir, dst_dir, ratio, seed=42):
    random.seed(seed)

    # images 디렉토리 기준 파일명 리스트
    images_dir = os.path.join(src_dir, 'images')
    file_names = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    file_names.sort()

    sample_size = int(len(file_names) * ratio)
    sampled_files = random.sample(file_names, sample_size)

    print(f"[{src_dir}]에서 {ratio*100:.0f}% → {sample_size}개 샘플링")

    # dst 하위 폴더 생성
    for subdir in ['images', 'labels', 'gt']:
        os.makedirs(os.path.join(dst_dir, subdir), exist_ok=True)

    for file_name in sampled_files:
        stem = os.path.splitext(file_name)[0]
        paths = {
            'images': (f"{stem}.png", 'png'),
            'labels': (f"{stem}.txt", 'txt'),
            'gt':     (f"{stem}.png", 'png'),
        }

        for subdir, (filename, _) in paths.items():
            src_path = os.path.join(src_dir, subdir, filename)
            dst_path = os.path.join(dst_dir, subdir, filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

# 설정
root_path = "../data/"      # 원본 경로, train & val 디렉토리가 먼저 존재해야 함
output_path = "../data/"   # 결과 저장 경로

# 샘플링 비율 설정
ratios = [0.01, 0.1, 0.3, 0.5]
splits = ['train', 'val']

# 반복 실행
for split in splits:
    for ratio in ratios:
        src = os.path.join(root_path, split)
        dst = os.path.join(output_path, f"{split}_{int(ratio*100)}")
        sample_and_save(src, dst, ratio)
