import os
import torch
import utils
import numpy as np
import torch.nn.functional as F
from timm.utils import accuracy
from timm.models import create_model
from sklearn.metrics import roc_auc_score

def load_model(args):
    model = create_model(
        "vit_base_patch16_224",
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
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
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Train: Epoch [{epoch+1}/{args.epochs}]'
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx>20:
                    break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            output = model(input)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.size(0))
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.size(0))
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def evaluate_task(self, model, data_loader, device, task_id, class_mask, args):
        """
        한 task에 대해 evaluation을 수행합니다.
        """
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f'Test: Task {task_id+1}'
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx>20:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.update(Loss=loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.size(0))
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.size(0))
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} Loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.meters['Acc@1'],
            top5=metric_logger.meters['Acc@5'],
            losses=metric_logger.meters['Loss']
        ))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def evaluate_ood(self, model, id_loader, ood_loader, device, args):
        """
        OOD 평가: 모델 출력 logits에서 softmax의 최대값을 id_score로,
        ood_score = 1 - id_score로 정의한 후, ID와 OOD 데이터를 구분하는 성능(예: AUROC 등)을 계산합니다.
        
        Args:
            id_loader: incremental 학습 시의 ID validation 데이터 로더.
            ood_loader: OOD 평가용 데이터 로더 (예: MNIST, 모든 라벨이 unknown_class로 변환된).
            args.unknown_class: OOD 데이터의 라벨 (예: 10).
            args.ood_threshold: OOD detection 임계값.
        """
        # ID 데이터 평가
        model.eval()
        all_id_logits = []
        all_id_targets = []
        with torch.no_grad():
            for inputs, targets in id_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_id_logits.append(outputs)
                all_id_targets.append(targets.cpu())
        all_id_logits = torch.cat(all_id_logits, dim=0)
        all_id_targets = torch.cat(all_id_targets, dim=0).to(device)
        id_softmax = F.softmax(all_id_logits, dim=1)
        max_softmax, _ = torch.max(id_softmax, dim=1)

        id_acc = accuracy(all_id_logits, all_id_targets, topk=(1,))[0].item()

        # OOD 데이터 평가
        all_ood_logits = []
        all_ood_targets = []
        with torch.no_grad():
            for inputs, targets in ood_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                all_ood_logits.append(outputs)
                all_ood_targets.append(targets.cpu())
        all_ood_logits = torch.cat(all_ood_logits, dim=0)
        all_ood_targets = torch.cat(all_ood_targets, dim=0)
        ood_softmax = F.softmax(all_ood_logits, dim=1)
        max_softmax_ood, _ = torch.max(ood_softmax, dim=1)
        # OOD score: 낮은 max softmax이면 OOD로 판단하기 위해 1 - max_softmax 사용
        ood_scores = 1 - max_softmax_ood

        # OOD 예측: ood_score가 threshold 이상이면 OOD로 예측 (즉, label = args.unknown_class)
        ood_preds = (ood_scores >= args.ood_threshold).long()
        # 실제 OOD 데이터의 모든 target은 args.unknown_class로 설정되어 있으므로, ood detection accuracy는
        # ood_preds가 1인 비율
        ood_acc = np.mean(ood_preds.cpu().numpy() == 1)

        # AUROC 계산: ID는 positive (1), OOD는 negative (0)
        id_binary = np.ones(all_id_logits.shape[0], dtype=np.int32)
        ood_binary = np.zeros(all_ood_logits.shape[0], dtype=np.int32)
        binary_labels = np.concatenate([id_binary, ood_binary])
        combined_scores = torch.cat([max_softmax, max_softmax_ood]).cpu().numpy()
        try:
            auc_roc = roc_auc_score(binary_labels, combined_scores)
        except Exception:
            auc_roc = 0.0

        print(f"OOD Evaluation: ID Acc: {id_acc:.3f}, OOD Acc: {ood_acc:.3f}, AUROC: {auc_roc:.3f}")
        return id_acc, ood_acc, auc_roc
    
    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for t in range(task_id+1):
            test_stats = self.evaluate_task(model, data_loader[t]['val'], device, t, class_mask, args)

            acc_matrix[t, task_id] = test_stats['Acc@1']

        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        result_str = "[Average accuracy till task{}] A_last: {:.4f} A_avg: {:.4f}".format(task_id+1, A_last, A_avg)
        
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        
        print(result_str)
        if args.verbose:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            print(result)



    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        """
        전체 incremental learning 과정을 수행합니다.
        각 task에 대해 지정된 epoch만큼 fine-tuning 후,
        지금까지의 task에 대해 평가합니다.
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"\n--- Training on Task {task_id+1}/{args.num_tasks} ---")
            for epoch in range(args.epochs):
                train_stats = self.train_one_epoch(model, criterion, data_loader[task_id]['train'], optimizer, device, epoch, args)
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)

            print(f"\n--- Testing on Task {task_id+1}/{args.num_tasks} ---")
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            print(f"Task {task_id+1} evaluation completed.")

            if args.output_dir and utils.is_main_process():
                checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
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
        
        if args.ood_dataset:
            print("\n--- OOD Evaluation ---")
            all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader])
            id_loader = torch.utils.data.DataLoader(all_id_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem)
            
            ood_loader = data_loader[-1]['ood']
            self.evaluate_ood(model, id_loader, ood_loader, device, args)
            print("OOD evaluation completed.")