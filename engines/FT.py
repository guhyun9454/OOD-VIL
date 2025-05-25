import os
import torch
import time
import datetime
import numpy as np
import torch.nn.functional as F
from timm.utils import accuracy
from timm.models import create_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from utils import save_accuracy_heatmap, save_anomaly_histogram, save_confusion_matrix_plot, save_logits_statistics
from continual_datasets.dataset_utils import RandomSampleWrapper  
import matplotlib.pyplot as plt

def load_model(args):
    model = create_model(
        "vit_base_patch16_224",
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
    
    # Linear probing 모드: backbone freeze, classifier만 학습
    if args.linear_probing:
        # 모든 파라미터를 freeze
        for param in model.parameters():
            param.requires_grad = False
        
        # classifier(head)만 학습 가능하게 설정
        if hasattr(model, 'head') and model.head is not None:
            for param in model.head.parameters():
                param.requires_grad = True
            print(f"Classifier layer만 학습 가능하게 설정되었습니다. (파라미터 수: {sum(p.numel() for p in model.head.parameters())})")
        else:
            print("Warning: 모델에 'head' 속성이 없거나 None입니다.")
    
    return model

class Engine:
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        """
        Args:
            model: 학습할 모델.
            device: torch.device 객체.
            class_mask: task별 클래스 마스크 (incremental scenario에 따른).
            domain_list: 각 task의 도메인 정보.
            args: argparse에서 전달된 인자 (num_tasks, epochs, print_freq 등 포함).
        """
        self.model = model
        self.device = device
        self.args = args
        self.class_mask = class_mask
        self.domain_list = domain_list
        self.num_tasks = args.num_tasks

    def train_one_epoch(self, model, criterion, data_loader, optimizer, device, epoch, args):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if self.args.develop and batch_idx > 20: break

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            batch_size = inputs.size(0)
            
            total_loss += loss.item() * batch_size
            total_acc += acc1.item() * batch_size
            total_samples += batch_size

            if batch_idx % args.print_freq == 0:
                current_avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
                current_avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_idx}/{len(data_loader)}]: "
                    f"Loss = {loss.item():.4f}, Acc@1 = {acc1.item():.2f}, "
                    f"Running Avg Loss = {current_avg_loss:.4f}, Running Avg Acc = {current_avg_acc:.2f}")

        epoch_avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_avg_acc = total_acc / total_samples if total_samples > 0 else 0.0
        return epoch_avg_loss, epoch_avg_acc

    def evaluate_task(self, model, data_loader, device, task_id, class_mask, args):
        """
        한 task에 대해 evaluation을 수행하며, 매 print_freq 배치마다 중간 결과를 출력합니다.
        """
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if args.develop and batch_idx > 20: break

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                batch_size = inputs.size(0)
                
                total_acc += acc1.item() * batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if batch_idx % args.print_freq == 0:
                    running_avg_loss = total_loss / total_samples
                    running_avg_acc = total_acc / total_samples
                    print(f"Task {task_id+1}, Batch [{batch_idx}/{len(data_loader)}]: "
                        f"Running Avg Loss = {running_avg_loss:.2f}, Running Avg Acc@1 = {running_avg_acc:.2f}")

        avg_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.2f} | Final Avg Acc@1 = {avg_acc:.2f}")
        return avg_acc

    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()
        
        # OOD detection method 선택 (MSP, ENERGY, KL 또는 ALL)
        ood_method = args.ood_method.upper()
        
        def MSP(logits):
            return F.softmax(logits, dim=1).max(dim=1)[0]
        
        def ENERGY(logits):
            return torch.logsumexp(logits, dim=1)
        
        def KL(logits):
            uniform = torch.ones_like(logits) / logits.shape[-1]
            return F.cross_entropy(logits, uniform, reduction='none')
        
        # 데이터셋 샘플 수 맞추기
        id_size = len(id_datasets)
        ood_size = len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            min_size = 1000
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")
        
        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset
        
        aligned_id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        aligned_ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # ID 및 OOD 데이터의 로짓
        id_logits_list = []
        ood_logits_list = []
        
        with torch.no_grad():
            # ID 데이터 처리
            for inputs, _ in aligned_id_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                id_logits_list.append(logits)
            
            # OOD 데이터 처리
            for inputs, _ in aligned_ood_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                ood_logits_list.append(logits)
        
        # ID 및 OOD 데이터의 로짓 합치기
        id_logits = torch.cat(id_logits_list, dim=0)
        ood_logits = torch.cat(ood_logits_list, dim=0)
        
        # Logits 통계 시각화 및 저장
        if args.save:
            stat_path = save_logits_statistics(id_logits, ood_logits, args, task_id if task_id is not None else 0)
            if args.wandb:
                import wandb
                wandb.log({f"Logits Statistics TASK {task_id}": wandb.Image(stat_path)})
        
        # binary_labels: 0 = OOD, 1 = ID
        binary_labels = np.concatenate([np.ones(id_logits.shape[0]),
                                       np.zeros(ood_logits.shape[0])])
        
        # 실행할 방법들 결정
        if ood_method == "ALL":
            methods = ["MSP", "ENERGY", "KL"]
        else:
            methods = [ood_method]
        
        results = {}
        
        # 각 방법에 대해 평가 실행
        for method in methods:
            if method == "MSP":
                id_scores = MSP(id_logits)
                ood_scores = MSP(ood_logits)
            elif method == "ENERGY":
                id_scores = ENERGY(id_logits)
                ood_scores = ENERGY(ood_logits)
            elif method == "KL":
                id_scores = KL(id_logits)
                ood_scores = KL(ood_logits)
            
            # anomaly score 히스토그램 저장 (verbose 모드)
            if args.verbose or args.wandb:
                hist_path = save_anomaly_histogram(id_scores.cpu().numpy(), ood_scores.cpu().numpy(), args, suffix=method.lower(), task_id=task_id)
                if args.wandb:
                    import wandb
                    wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})
            
            all_scores = torch.cat([id_scores, ood_scores], dim=0).cpu().numpy()
            
            # ROC 및 필요한 지표만 계산
            from sklearn import metrics
            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]
            
            print(f"[{method}]: evaluating metrics...")
            print(f"AUROC: {auroc * 100:.2f}%, FPR@TPR95: {fpr_at_tpr95 * 100:.2f}%")
            if args.wandb:
                import wandb
                wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id})
            
            results[method] = {
                "auroc": auroc,
                "fpr_at_tpr95": fpr_at_tpr95,
                "scores": all_scores
            }
        
        return results
    
    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for t in range(task_id+1):
            acc_matrix[t, task_id] = self.evaluate_task(model, data_loader[t]['val'], device, t, class_mask, args)

        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id+1, A_last, A_avg)
        
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        else:
            forgetting = 0

        if args.wandb:
            import wandb
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})
        
        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb
                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})
        
        return {"Acc@1": A_last}

    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        """
        전체 incremental learning 과정을 수행합니다.
        각 task에 대해 지정된 epoch만큼 fine-tuning 후,
        지금까지의 task에 대해 평가합니다.
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")
            train_start = time.time()
            for epoch in range(args.epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch(model, criterion, data_loader[task_id]['train'], optimizer, device, epoch, args)
                epoch_duration = time.time() - epoch_start  
                print(f"Epoch [{epoch+1}/{args.epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}")

                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)
            train_duration = time.time() - train_start
            print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")
            print(f'{f"Testing on Task {task_id+1}/{args.num_tasks}":=^60}')
            eval_start = time.time()
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            eval_duration = time.time() - eval_start
            print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")

            if args.ood_dataset:
                print(f"{f'OOD Evaluation':=^60}")
                ood_start = time.time()
                # 현재 task까지의 ID 데이터셋만 사용
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]['val'].dataset for t in range(task_id+1)])
                ood_loader = data_loader[-1]['ood']
                self.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation after Task {task_id+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
                
            # 추가: 일반 추론 시의 logits 통계 저장 - 제거
            if args.save:
                checkpoint_dir = os.path.join(args.save, 'checkpoint')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'task{}_checkpoint.pth'.format(task_id+1))
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'linear_probing': args.linear_probing,
                }
                with open(checkpoint_path, 'wb') as f:
                    torch.save(checkpoint, f)
                print(f"Saved checkpoint for task {task_id+1} at {checkpoint_path}")

    def load_checkpoint(self, model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'체크포인트를 찾을 수 없습니다: {checkpoint_path}')
        
        print(f'체크포인트를 로드합니다: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model'])

        return model