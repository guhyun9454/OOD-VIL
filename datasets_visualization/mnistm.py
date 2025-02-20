import os
import torch
import numpy as np
from PIL import Image

# 불러올 .pt 파일 경로 (원하는 파일로 변경 가능)
pt_file = '/local_datasets/MNIST-M/processed/mnist_m_train.pt'

# 데이터셋 불러오기: (data, targets) 형태로 저장되어 있다고 가정합니다.
data, targets = torch.load(pt_file)

# 결과 이미지 저장 폴더 생성
output_dir = '/data/guhyun9454/downloads/mnist_m_images'
os.makedirs(output_dir, exist_ok=True)

# 클래스별 서브 폴더 생성 (0부터 9까지)
for cls in range(10):
    class_dir = os.path.join(output_dir, f'class_{cls}')
    os.makedirs(class_dir, exist_ok=True)

# 각 이미지에 대해 해당 클래스 폴더에 저장
for idx, (img_tensor, label) in enumerate(zip(data, targets)):
    # tensor에서 numpy array로 변환; squeeze()로 불필요한 차원 제거
    img_np = img_tensor.squeeze().numpy()
    
    # 만약 이미지가 (채널, H, W) 형태라면 (H, W, 채널)로 변환
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    
    # 만약 이미지가 2차원(그레이스케일)이라면 3채널로 복제하여 RGB 이미지로 변환
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    
    # numpy array를 PIL 이미지로 변환 (uint8 타입으로 변환)
    img = Image.fromarray(img_np.astype(np.uint8), mode='RGB')
    
    # 이미지 저장 (예: mnist_m_images/class_3/img_57.png)
    save_path = os.path.join(output_dir, f'class_{int(label)}', f'img_{idx}.png')
    img.save(save_path)

print("이미지 저장 완료! 결과는 '{}' 폴더에서 확인하세요.".format(output_dir))