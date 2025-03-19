import numpy as np
import torch
from timm.utils import accuracy 
import utils  
from timm.models import create_model

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

    def evaluate(self, model, data_loader, device, task_id, class_mask, args):
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

    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix, args):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting, Backward 등의 지표를 계산하여 출력합니다.
        """
        stat_matrix = np.zeros((3, args.num_tasks))
        for t in range(task_id+1):
            test_stats = self.evaluate(model, data_loader[t]['val'], device, t, class_mask, args)
            stat_matrix[0, t] = test_stats['Acc@1']
            stat_matrix[1, t] = test_stats['Acc@5']
            stat_matrix[2, t] = test_stats['Loss']
            acc_matrix[t, task_id] = test_stats['Acc@1']

        avg_stat = np.sum(stat_matrix, axis=1) / (task_id + 1)
        A_last = avg_stat[0]
        A_avg = np.mean(acc_matrix[np.triu_indices(task_id+1)])
        
        result_str = "[Average accuracy till task{}] A_last: {:.4f} A_avg: {:.4f}".format(task_id+1, A_last, A_avg)
        
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            diagonal = np.diag(acc_matrix)
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
            result_str += " Forgetting: {:.4f} Backward: {:.4f}".format(forgetting, backward)
        
        print(result_str)
        return test_stats


    def train_and_evaluate(self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask, args):
        """
        전체 incremental learning 과정을 수행합니다.
        각 task에 대해 지정된 epoch만큼 fine-tuning 후,
        지금까지의 task에 대해 평가합니다.
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        for task_id in range(args.num_tasks):
            print(f"\n--- Training on Task {task_id+1}/{args.num_tasks} ---")
            # 각 task별 optimizer 재설정은 FT baseline에서는 재사용해도 무방합니다.
            for epoch in range(args.epochs):
                train_stats = self.train_one_epoch(model, criterion, data_loader[task_id]['train'], optimizer, device, epoch, args)
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)
            # task 학습 후, 지금까지의 모든 task에 대해 평가합니다.
            print(f"\n--- Testing on Task {task_id+1}/{args.num_tasks} ---")
            stats = self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            print(f"Task {task_id+1} evaluation completed.")
