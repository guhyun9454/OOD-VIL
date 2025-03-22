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
from utils import save_accuracy_heatmap, save_anomaly_histogram, save_confusion_matrix_plot
from continual_datasets.dataset_utils import RandomSampleWrapper  

def load_model(args):
    model = create_model(
        "vit_base_patch16_224",
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
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

    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args):
        """
        Revised OOD evaluation:
        - Align ID and OOD datasets to the same number of samples.
        - For each sample, compute softmax outputs and determine prediction: if max_softmax < 0.5, predict as unknown (args.unknown_class), otherwise use argmax.
        - Combine predictions from ID and OOD data to build a (num_classes+1) x (num_classes+1) confusion matrix.
        - Compute AUROC using binary labels (0 for ID, 1 for OOD) and ood scores (1 - max_softmax).
        """

        # Align datasets: use the smaller dataset size for both ID and OOD
        id_size = len(id_datasets)
        ood_size = len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.verbose: print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

        # Use RandomSampleWrapper if dataset size is larger than min_size
        if id_size > min_size:
            id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed)
        else:
            id_dataset_aligned = id_datasets
        if ood_size > min_size:
            ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed)
        else:
            ood_dataset_aligned = ood_dataset

        aligned_id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        aligned_ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_true = []
        all_pred = []
        binary_labels = []  # 0 for ID, 1 for OOD
        ood_scores_all = []  # 1 - max_softmax for each sample
        id_true_list, id_pred_list = [], []
        ood_true_list, ood_pred_list = [], []


        model.eval()
        with torch.no_grad():
            # Process ID data
            for inputs, targets in aligned_id_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                softmax_outputs = F.softmax(outputs, dim=1)
                max_softmax, pred_class = torch.max(softmax_outputs, dim=1)
                # If max_softmax is below 0.5, predict as unknown
                pred = torch.where(max_softmax < args.ood_threshold, torch.full_like(pred_class, args.num_classes), pred_class)
                all_true.extend(targets.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                binary_labels.extend([0] * inputs.size(0))
                ood_scores_all.extend((1 - max_softmax).cpu().numpy())
                id_true_list.extend(targets.cpu().numpy())
                id_pred_list.extend(pred.cpu().numpy())

            # Process OOD data
            for inputs, targets in aligned_ood_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                softmax_outputs = F.softmax(outputs, dim=1)
                max_softmax, pred_class = torch.max(softmax_outputs, dim=1)
                pred = torch.where(max_softmax < args.ood_threshold, torch.full_like(pred_class, args.num_classes), pred_class)
                # For OOD, true label should be set to unknown_class
                true_labels = torch.full_like(targets, fill_value=args.num_classes)
                all_true.extend(true_labels.cpu().numpy())
                all_pred.extend(pred.cpu().numpy())
                binary_labels.extend([1] * inputs.size(0))
                ood_scores_all.extend((1 - max_softmax).cpu().numpy())
                ood_true_list.extend(true_labels.cpu().numpy())
                ood_pred_list.extend(pred.cpu().numpy())

        # Build confusion matrix: labels from 0 to (num_classes - 1) and unknown_class
        labels = list(range(args.num_classes+1)) 
        conf_matrix = confusion_matrix(all_true, all_pred, labels=labels)

            # Compute ID and OOD accuracy separately
        id_total = len(id_true_list)
        ood_total = len(ood_true_list)
        id_correct = sum(1 for t, p in zip(id_true_list, id_pred_list) if t == p)
        ood_correct = sum(1 for t, p in zip(ood_true_list, ood_pred_list) if t == p)
        acc_id = id_correct / id_total if id_total > 0 else 0.0
        acc_ood = ood_correct / ood_total if ood_total > 0 else 0.0
        h_score = 2 * acc_id * acc_ood / (acc_id + acc_ood) 

        # Compute AUROC for OOD detection using binary labels and ood scores
        try:
            auroc = roc_auc_score(binary_labels, ood_scores_all)
        except Exception as e:
            print("AUROC computation error:", e)
            auroc = 0.0
        print(f"OOD Evaluation: AUROC: {auroc * 100:.2f}, A_id: {acc_id * 100:.2f}, A_ood: {acc_ood * 100:.2f}  H_score: {h_score * 100:.2f}")

        if args.verbose:
            save_confusion_matrix_plot(conf_matrix, labels, args)
            id_scores = [score for score, label in zip(ood_scores_all, binary_labels) if label == 0]
            ood_scores = [score for score, label in zip(ood_scores_all, binary_labels) if label == 1]
            save_anomaly_histogram(id_scores, ood_scores, args)

        return conf_matrix, auroc
    
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
        
        print(result_str)
        if args.verbose:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            save_accuracy_heatmap(result, task_id, args)

        if task_id == args.num_tasks - 1 and args.ood_dataset:
            print(f"{'OOD Evaluation':=^60}")
            ood_start = time.time()
            all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader])
            
            ood_loader = data_loader[-1]['ood']
            self.evaluate_ood(model, all_id_datasets, ood_loader, device, args)
            ood_duration = time.time() - ood_start
            print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")

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

            if args.save:
                checkpoint_dir = os.path.join(args.save, 'checkpoint')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, 'task{}_checkpoint.pth'.format(task_id+1))
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                with open(checkpoint_path, 'wb') as f:
                    torch.save(checkpoint, f)
                print(f"Saved checkpoint for task {task_id+1} at {checkpoint_path}")