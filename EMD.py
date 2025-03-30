import torch
import numpy as np
import clip
import ot  # POT 라이브러리 (pip install POT)
from torch.utils.data import DataLoader
from continual_datasets.dataset_utils import get_dataset, build_transform
import argparse
from tqdm import tqdm
import timm

def compute_prototype_arrays(features, labels):
    unique_labels = np.unique(labels)
    prototypes_list = []
    weights_list = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        prototypes_list.append(np.mean(features[idx], axis=0))
        weights_list.append(len(idx) / len(labels))
    prototypes_array = np.vstack(prototypes_list)
    weights_array = np.array(weights_list)
    return prototypes_array, weights_array

def compute_cost_matrix(prototypes_A, prototypes_B):
    nA = prototypes_A.shape[0]
    nB = prototypes_B.shape[0]
    cost_matrix = np.zeros((nA, nB))
    for i in range(nA):
        for j in range(nB):
            cost_matrix[i, j] = np.linalg.norm(prototypes_A[i] - prototypes_B[j])
    return cost_matrix

def compute_domain_similarity(features_A, labels_A, features_B, labels_B, alpha=0.01):
    prototypes_A, weights_A = compute_prototype_arrays(features_A, labels_A)
    prototypes_B, weights_B = compute_prototype_arrays(features_B, labels_B)
    cost_matrix = compute_cost_matrix(prototypes_A, prototypes_B)
    transport_plan = ot.emd(weights_A, weights_B, cost_matrix)
    emd_value = np.sum(transport_plan * cost_matrix)
    domain_similarity = np.exp(-alpha * emd_value)
    return domain_similarity, emd_value

def extract_features_clip(dataset, batch_size=32, device="cuda"):

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    
    model.eval()  
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader):
            imgs = imgs.to(device)
            feat = model.encode_image(imgs)
            features.append(feat.detach().cpu().numpy())  
            labels.extend(lbls.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def extract_features_vit(dataset, batch_size=32, device="cuda"):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader):
            imgs = imgs.to(device)
            feat = model(imgs)
            features.append(feat.detach().cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels


def process_two_datasets(dataset1_name, dataset2_name, args):
    # 데이터 전처리 transform 구성
    transform_train = build_transform(is_train=True, args=args)
    transform_val = build_transform(is_train=False, args=args)
    mode = "joint"  # joint 모드

    # 각 데이터셋 로드 (여기서는 validation 데이터 사용)
    print(f"데이터셋 '{dataset1_name}' 로드 중...")
    ds1_train, ds1_val = get_dataset(dataset1_name, transform_train, transform_val, mode, args)
    print(f"데이터셋 '{dataset2_name}' 로드 중...")
    ds2_train, ds2_val = get_dataset(dataset2_name, transform_train, transform_val, mode, args)
    
    dataset1 = ds1_val
    dataset2 = ds2_val

    print(f"'{dataset1_name}' 데이터셋 크기: {len(dataset1)}")
    print(f"'{dataset2_name}' 데이터셋 크기: {len(dataset2)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 특징 추출
    print(f"'{dataset1_name}' 데이터셋 인코딩 시작...")
    features1, labels1 = extract_features_vit(dataset1, batch_size=512, device=device)
    print(f"'{dataset2_name}' 데이터셋 인코딩 시작...")
    features2, labels2 = extract_features_vit(dataset2, batch_size=512, device=device)

    print(f"'{dataset1_name}' 특징 shape: {features1.shape}")
    print(f"'{dataset2_name}' 특징 shape: {features2.shape}")

    # EMD 기반 도메인 유사도 계산
    similarity, emd_value = compute_domain_similarity(features1, labels1, features2, labels2)
    print("도메인 유사도:", similarity)
    print("EMD 값:", emd_value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", type=str, default="MNIST", help="첫 번째 데이터셋 이름")
    parser.add_argument("--dataset2", type=str, default="SVHN", help="두 번째 데이터셋 이름")
    parser.add_argument("--data_path", type=str, default="/local_datasets/", help="데이터셋 경로")
    args = parser.parse_args()
    args.verbose = True

    process_two_datasets(args.dataset1, args.dataset2, args)

    
