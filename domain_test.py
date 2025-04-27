import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 모듈 추가
from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import set_data_config

class FeatureExtractorWithHead(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_extractor.num_features, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

def parse_args():
    parser = argparse.ArgumentParser(description='도메인 간 테스트')
    
    # 데이터셋 관련 인자
    parser.add_argument('--dataset', type=str, default='iDigits', help='사용할 데이터셋')
    parser.add_argument('--IL_mode', type=str, default='dil', help='학습 모드 (dil 모드 사용)')
    parser.add_argument('--num_tasks', type=int, default=4, help='태스크 수 (도메인 수)')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
    parser.add_argument('--img_size', type=int, default=224, help='입력 이미지 크기')
    parser.add_argument('--shuffle', action='store_true', help='클래스 셔플 여부')
    parser.add_argument('--verbose', action='store_true', help='상세 로그 출력 여부')
    parser.add_argument('--develop_tasks', action='store_true', help='개발용 태스크 출력 여부')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='사용할 timm 모델명')
    parser.add_argument('--develop', action='store_true')
    parser.add_argument('--data_path', type=str, default='/local_datasets/', help='데이터 경로')
    
    # 학습 관련 인자
    parser.add_argument('--epochs', type=int, default=5, help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--save_dir', type=str, default='./domain_test', help='결과 저장 디렉토리')
    
    # 시각화 모드 관련 인자
    parser.add_argument('--visualize_only', type=str, default='', help='.npy 파일 경로를 입력하면 시각화만 수행')
    parser.add_argument('--domain_names', type=str, default='', help='시각화 시 사용할 도메인 이름 (콤마로 구분)')
    
    args = parser.parse_args()
    return args

def train(model, train_loader, criterion, optimizer, device, args):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i,(inputs, targets) in enumerate(train_loader):
        if args.develop and i > 10:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc

def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc

def visualize_accuracy_matrix(accuracy_matrix, domain_list, args):
    """
    정확도 매트릭스를 시각화하고 저장하는 함수
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(accuracy_matrix, cmap='Reds', interpolation='nearest', vmin=0, vmax=100)
    plt.colorbar(label='Accuracy (%)')
    plt.title(f'Domain Cross-Evaluation Accuracy Matrix - {args.dataset}')
    plt.xlabel('Evaluation Domain')
    plt.ylabel('Training Domain')
    
    # x축과 y축에 도메인 이름 표시
    plt.xticks(np.arange(len(domain_list)), domain_list)
    plt.yticks(np.arange(len(domain_list)), domain_list)
    
    # 각 셀에 정확도 값 표시
    for i in range(len(domain_list)):
        for j in range(len(domain_list)):
            plt.text(j, i, f'{accuracy_matrix[i, j]:.1f}',
                     ha='center', va='center', 
                     color='white' if accuracy_matrix[i, j] < 70 else 'black')
    
    plt.tight_layout()
    
    # 결과 저장 디렉토리 확인 및 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 파일 저장
    save_path = os.path.join(args.save_dir, f'{args.dataset}.png')
    plt.savefig(save_path)
    print(f"시각화 결과가 {save_path}에 저장되었습니다.")
    plt.show()

def main():
    args = parse_args()
    
    # 시각화 모드인 경우
    if args.visualize_only:
        print(f".npy 파일에서 정확도 매트릭스를 로드하여 시각화합니다: {args.visualize_only}")
        
        # .npy 파일 로드
        try:
            accuracy_matrix = np.load(args.visualize_only)
            print(f"정확도 매트릭스 로드 완료 (크기: {accuracy_matrix.shape})")
            
            # 도메인 이름 설정
            if args.domain_names:
                domain_list = args.domain_names.split(',')
                if len(domain_list) != accuracy_matrix.shape[0]:
                    print(f"경고: 도메인 이름 개수({len(domain_list)})가 매트릭스 크기({accuracy_matrix.shape[0]})와 일치하지 않습니다.")
                    domain_list = [f"D{i}" for i in range(accuracy_matrix.shape[0])]
            else:
                # 도메인 이름이 없으면 기본값 사용
                domain_list = [f"D{i}" for i in range(accuracy_matrix.shape[0])]
            
            # 정확도 매트릭스 시각화
            visualize_accuracy_matrix(accuracy_matrix, domain_list, args)
            return
        except Exception as e:
            print(f"오류: {e}")
            return
    
    # 이 아래는 기존 학습 및 평가 코드
    set_data_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 결과 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 데이터 로더 생성
    dataloaders, class_mask, domain_list = build_continual_dataloader(args)
    num_domains = len(dataloaders)
    print(f"생성된 도메인 수: {num_domains}")
    print(f"도메인 리스트: {domain_list}")
    
    # 각 도메인별 결과를 저장할 매트릭스 초기화
    accuracy_matrix = np.zeros((num_domains, num_domains))
    
    # 각 도메인 학습 시간을 저장할 리스트
    domain_training_times = []
    
    # 전체 학습 시작 시간 기록
    total_start_time = time.time()
    
    # 특징 추출기 미리 로드 (모든 도메인에서 재사용)
    feature_extractor = timm.create_model(args.model, pretrained=True, num_classes=0)
    # 특징 추출기 freeze
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    # 모델 아키텍처 준비 및 특징 추출기 freeze
    for train_domain_idx in range(num_domains):
        print(f"\n{'='*50}")
        print(f"도메인 {train_domain_idx} ({domain_list[train_domain_idx]}) 학습 시작")
        print(f"{'='*50}")
        
        # 도메인 학습 시작 시간 기록
        domain_start_time = time.time()
        
        # 특징 추출기에 새 분류 헤드 연결
        num_classes = args.num_classes
        model = FeatureExtractorWithHead(feature_extractor, num_classes)
        model = model.to(device)
        
        # 손실 함수 및 옵티마이저 설정 (분류기만 학습)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
        
        # 현재 도메인으로 학습
        train_loader = dataloaders[train_domain_idx]['train']
        
        # 학습 시작
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, args)
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, 시간: {epoch_time:.2f}초')
        
        # 모든 도메인에 대해 평가
        print(f"\n{'-'*50}")
        print(f"도메인 {train_domain_idx} ({domain_list[train_domain_idx]}) 모델 평가 결과:")
        print(f"{'-'*50}")
        
        for val_domain_idx in range(num_domains):
            if args.dataset == 'CORe50':
                val_loader = dataloaders[val_domain_idx]['test']
            else:
                val_loader = dataloaders[val_domain_idx]['val']
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            accuracy_matrix[train_domain_idx, val_domain_idx] = val_acc
            
            print(f"도메인 {val_domain_idx} ({domain_list[val_domain_idx]}) - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 도메인 학습 종료 시간 및 소요 시간 계산
        domain_end_time = time.time()
        domain_time = domain_end_time - domain_start_time
        domain_training_times.append(domain_time)
        print(f"도메인 {train_domain_idx} ({domain_list[train_domain_idx]}) 학습 시간: {domain_time:.2f}초")
    
    # 전체 학습 종료 시간 및 총 소요 시간 계산
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 결과 시각화 및 저장
    visualize_accuracy_matrix(accuracy_matrix, domain_list, args)
    
    # 결과 저장
    np.save(os.path.join(args.save_dir, f'{args.dataset}_accuracy_matrix.npy'), accuracy_matrix)
    
    print(f"\n결과가 {args.save_dir} 디렉토리에 저장되었습니다.")
    
    # 결과 분석
    diagonal_avg = np.mean(np.diag(accuracy_matrix))
    off_diagonal_avg = (np.sum(accuracy_matrix) - np.sum(np.diag(accuracy_matrix))) / (num_domains * (num_domains - 1))
    
    print(f"\n{'='*50}")
    print(f"결과 분석:")
    print(f"{'='*50}")
    print(f"같은 도메인 평균 정확도: {diagonal_avg:.2f}%")
    print(f"다른 도메인 평균 정확도: {off_diagonal_avg:.2f}%")
    print(f"도메인 차이 (같은 도메인 - 다른 도메인): {diagonal_avg - off_diagonal_avg:.2f}%")
    
    # 시간 측정 결과 출력
    print(f"\n{'='*50}")
    print(f"시간 측정 결과:")
    print(f"{'='*50}")
    for i, domain_time in enumerate(domain_training_times):
        print(f"도메인 {i} ({domain_list[i]}) 학습 시간: {domain_time:.2f}초")
    print(f"총 학습 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")

if __name__ == '__main__':
    main() 