import math
import sys
import os
import time
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from timm.models import create_model
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix
from continual_datasets.dataset_utils import RandomSampleWrapper
from utils import save_accuracy_heatmap, save_logits_statistics, save_anomaly_histogram

def load_model(args):
    if args.dataset == 'CORe50':
        init_ICON_CORe50_args(args)
        print("CORe50 dataset args loaded")
    elif args.dataset == 'iDigits':
        init_ICON_iDigits_args(args)
        print("iDigits dataset args loaded")
    elif args.dataset == 'DomainNet':
        init_ICON_DomainNet_args(args)
        print("DomainNet dataset args loaded")
    else:
        init_ICON_default_args(args)
        print("Default dataset args loaded")

    model = create_model(
        "vit_base_patch16_224_ICON",
        pretrained=True,
        num_classes=args.num_classes,  #10
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        adapt_blocks=args.adapt_blocks,  #[0, 1, 2, 3, 4]
    )
    for n, p in model.named_parameters():
        p.requires_grad = False
        if 'adapter' in n:
            p.requires_grad = True
        if 'head' in n:
            p.requires_grad = True
    return model

class Engine():
    def __init__(self, model=None, device=None, class_mask=[], domain_list=[], args=None):
        self.current_task = 0
        self.current_classes = []
        #! distillation
        self.class_group_num = 5
        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        
        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.distill_head = None
        self.model = model
        
        self.num_classes = max([item for mask in class_mask for item in mask]) + 1
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args = args
        
        self.class_mask = class_mask
        self.domain_list = domain_list

        self.task_type = "initial"
        self.args = args
        
        self.adapter_vec = []
        self.task_type_list = []
        self.class_group_list = []
        self.adapter_vec_label = []
        self.device = device
        
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.num_classes, self.args.num_domains))
            self.label_train_count = np.zeros((self.args.num_classes))
            self.tanh = torch.nn.Tanh()
            
        self.cs = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def kl_div(self, p, q):
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q), dim=1))
        return kl
  
    def set_new_head(self, model, labels_to_be_added, task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id] * len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape  # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)
    
        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head
    
    def inference_acc(self, model, data_loader, device):
        print("Start detecting labels to be added...")
        accuracy_per_label = []
        correct_pred_per_label = [0 for i in range(len(self.current_classes))]
        num_instance_per_label = [0 for i in range(len(self.current_classes))]
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop:
                    if batch_idx > 200:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                if output.shape[-1] > self.num_classes:  # there are already added nodes till now
                    output, _, _ = self.get_max_label_logits(output, self.current_classes)  # get maximum value for each label
                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)
                
                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += sum(mask).item()
        for correct, num in zip(correct_pred_per_label, num_instance_per_label):
            accuracy_per_label.append(round(correct / num, 2))
        return accuracy_per_label
    
    def detect_labels_to_be_added(self, inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label, acc, thre in zip(self.current_classes, inference_acc, thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else:  # static threshold
            for label, acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self, vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs, dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs, dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self, diff_adapter, other):
        weights = []
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights)  # normalization
        return weights
    
    def train_one_epoch(self, model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model=None, args=None):
        model.train(set_training_mode)

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx > 20:
                    break
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input)  # (bs, class + n)
            distill_loss = 0
            if self.distill_head is not None:
                feature = model.forward_features(input)[:, 0]
                output_distill = self.distill_head(feature) 
                mask = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask)[0]
                m = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]), torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m]
                distill_loss = self.kl_div(output[:, distill_node_indices], output_distill[:, distill_node_indices])
               
            if output.shape[-1] > self.num_classes:
                output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]
                        output[:, added_class] = output[:, cur_node]
                output = output[:, :self.num_classes]       
                
            if class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)
           
            if self.args.CAST:
                if len(self.adapter_vec) > args.k:
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters
                    _, other = self.find_same_cluster_items(diff_adapter)
                    sim = 0
                    weights = self.calculate_l2_distance(diff_adapter, other)
                    for o, w in zip(other, weights):
                        if self.args.norm_cast:
                            sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter) * torch.norm(o))
                        else:
                            sim += w * torch.matmul(diff_adapter, o)
                    orth_loss = args.beta * torch.abs(sim)
                    if orth_loss > 0:
                        loss += orth_loss
                    
            if self.args.IC:
                if distill_loss > 0:
                    loss += distill_loss
           
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            optimizer.step()
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            if ema_model is not None:
                ema_model.update(model.get_adapter())
            
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def get_max_label_logits(self, output, class_mask, task_id=None, slice=True, target=None):
        for label in range(self.num_classes): 
            label_nodes = np.where(self.labels_in_head == label)[0]
            output[:, label], max_index = torch.max(output[:, label_nodes], dim=1)
        if slice:
            output = output[:, :self.num_classes]
        return output, 0, 0
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader, 
                 device, task_id=-1, class_mask=None, ema_model=None, args=None):
    
        if not self.current_classes or len(self.current_classes) == 0:
            self.current_classes = class_mask[task_id]

        criterion = torch.nn.CrossEntropyLoss()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)
        model.eval()
        correct_sum, total_sum = 0, 0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx > 20:
                        break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(input)
                output, correct, total = self.get_max_label_logits(output, class_mask[task_id], task_id=task_id, target=target, slice=True) 
                output_ema = [output.softmax(dim=1)]
                correct_sum += correct
                total_sum += total
                output = torch.stack(output_ema, dim=-1).max(dim=-1)[0]
                loss = criterion(output, target)
                
                # if self.args.d_threshold and self.current_task + 1 != self.args.num_tasks and self.current_task == task_id:
                #     label_correct, label_total = self.update_acc_per_label(label_correct, label_total, output, target)
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum/total_sum}")
                
            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total, decimals=3)
                # print(self.label_train_count)
                # print(self.acc_per_label)
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate_till_now(self, model: torch.nn.Module, data_loader, 
                          device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None):
        """
        현재까지의 모든 task에 대해 평가하고,
        A_last, A_avg, Forgetting 지표를 계산하여 출력합니다.
        """
        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                       device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)
            acc_matrix[i, task_id] = test_stats['Acc@1']
        
        A_i = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)
        
        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id+1, A_last, A_avg)


        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.2f}".format(forgetting)
        else:
            forgetting = 0
        
        if args.wandb:
            import wandb
            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting})
        
        print(result_str)
        if args.verbose:
            sub_matrix = acc_matrix[:task_id+1, :task_id+1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            save_accuracy_heatmap(result, task_id, args)
        
        return test_stats

    def flatten_parameters(self, modules):
        flattened_params = []
        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params)
        return torch.cat([param.view(-1) for param in flattened_params])
    
    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)
            self.kmeans = KMeans(n_clusters=k, n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)
    
    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args=None):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))
        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')        
        return model
    
    def pre_train_task(self, model, data_loader, device, task_id, args):
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id]) / self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]
        print(f"\n\nTASK : {task_id}")
        self.added_classes_in_cur_task = set()  
        if self.class_group_train_count[self.current_class_group] == 0:
            self.distill_head = None
        else:
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                thresholds = []
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    thresholds = self.args.gamma * (average_accs - inf_acc) / average_accs
                    thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                    thresholds = [round(t, 2) if t > self.args.thre else self.args.thre for t in thresholds]
                    print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
                if len(labels_to_be_added) > 0:
                    new_head = self.set_new_head(model, labels_to_be_added, task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)
        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad = False
        if task_id == 0:
            self.task_type_list.append("Initial")
            return model, optimizer
        prev_class = self.class_mask[task_id - 1]
        prev_domain = self.domain_list[task_id - 1]
        cur_class = self.class_mask[task_id]
        self.cur_domain = self.domain_list[task_id]
        if prev_class == cur_class:
            self.task_type = "DIL"
        else:
            self.task_type = "CIL"
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        return model, optimizer

    def post_train_task(self, model: torch.nn.Module, task_id=-1):
        self.class_group_train_count[self.current_class_group] += 1
        self.classifier_pool[self.current_class_group] = copy.deepcopy(model.head)
        for c in self.classifier_pool:
            if c is not None:
                for p in c.parameters():
                    p.requires_grad = False
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector = self.cur_adapters - self.prev_adapters
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()
                 
    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                           lr_scheduler, device: torch.device, class_mask=None, args=None):
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        ema_model = None
        for task_id in range(args.num_tasks):
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            if task_id == 1 and len(args.adapt_blocks) > 0:
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
            model, optimizer = self.pre_train_task(model, data_loader[task_id]['train'], device, task_id, args)
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args)
                train_stats = self.train_one_epoch(model=model, criterion=criterion, 
                                                   data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                                   device=device, epoch=epoch, max_norm=args.clip_grad, 
                                                   set_training_mode=True, task_id=task_id, class_mask=class_mask, ema_model=ema_model, args=args)
                if lr_scheduler:
                    lr_scheduler.step(epoch)
            self.post_train_task(model, task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1 
            test_stats = self.evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, ema_model=ema_model, args=args)
            
            if args.ood_dataset:
                print(f"{'OOD Evaluation':=^60}")
                ood_start = time.time()
                # ID 데이터셋은 모든 태스크의 검증 데이터(ICON의 예측 로직 적용)를 합침
                all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader[:task_id+1]])
                # ood_loader는 마지막 태스크에 추가된 ood 데이터 로더 사용
                ood_loader = data_loader[-1]['ood']
                self.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
            
            if args.save and utils.is_main_process():
                Path(os.path.join(args.save, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    # 추가: head 관련 정보 저장
                    'head_info': {
                        'labels_in_head': self.labels_in_head.tolist(),
                        'added_classes_in_cur_task': list(self.added_classes_in_cur_task),
                        'head_timestamps': self.head_timestamps.tolist(),
                        'num_classes': self.num_classes
                    }
                }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                utils.save_on_master(state_dict, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,}
            if args.save and utils.is_main_process():
                with open(os.path.join(args.save, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

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
                outputs = model(inputs)
                outputs, _, _ = self.get_max_label_logits(outputs, list(range(self.num_classes)), slice=True)
                id_logits_list.append(outputs)
            
            # OOD 데이터 처리
            for inputs, _ in aligned_ood_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs, _, _ = self.get_max_label_logits(outputs, list(range(self.num_classes)), slice=True)
                ood_logits_list.append(outputs)
        
        # ID 및 OOD 데이터의 로짓 합치기
        id_logits = torch.cat(id_logits_list, dim=0)
        ood_logits = torch.cat(ood_logits_list, dim=0)
        
        # Logits 통계 시각화 및 저장
        if args.save:
            save_logits_statistics(id_logits, ood_logits, args, task_id if task_id is not None else 0)
        
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
            if args.verbose:
                save_anomaly_histogram(id_scores.cpu().numpy(), ood_scores.cpu().numpy(), args, suffix=method.lower(), task_id=task_id)
            
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
                wandb.log({f"{method}_AUROC (↑)": auroc, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95})

            results[method] = {
                "auroc": auroc,
                "fpr_at_tpr95": fpr_at_tpr95,
                "scores": all_scores
            }
        
        return results

    def restore_head_from_checkpoint(self, model, checkpoint):
        if 'head_info' in checkpoint:
            head_info = checkpoint['head_info']
            self.labels_in_head = np.array(head_info['labels_in_head'])
            self.added_classes_in_cur_task = set(head_info['added_classes_in_cur_task'])
            self.head_timestamps = np.array(head_info['head_timestamps'])
            self.num_classes = head_info['num_classes']
            
            # 확장된 head를 가진 원본 모델과 동일한 크기의 새 head 생성
            orig_head_size = model.head.weight.shape[0]
            target_head_size = len(self.labels_in_head)
            
            if orig_head_size != target_head_size:
                print(f"Adjusting head size from {orig_head_size} to {target_head_size} to match checkpoint")
                in_features = model.head.in_features
                new_head = torch.nn.Linear(in_features, target_head_size)
                model.head = new_head
                
            return model
        else:
            print("Warning: No head information found in checkpoint. Loading model as is.")
            return model
            
    def load_checkpoint(self, model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'체크포인트를 찾을 수 없습니다: {checkpoint_path}')
        
        print(f'체크포인트를 로드합니다: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        
        # head 정보 복원
        model = self.restore_head_from_checkpoint(model, checkpoint)
        
        # 모델 파라미터 로드
        model.load_state_dict(checkpoint['model'])

        return model

def init_ICON_default_args(args):
    args.IC = True  #
    args.d_threshold = True  #
    args.gamma = 10.0  
    args.thre = 0 
    args.alpha = 1.0  

    args.CAST = True 
    args.beta = 0.01  
    args.k = 2  
    args.norm_cast = True 

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.ema_decay = 0.9999  
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 

    args.clip_grad = 0.0
    args.reinit_optimizer = True

def init_ICON_CORe50_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 3 #Number of Clusters
    args.alpha = 1
    args.beta = 0.05
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0 
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 

def init_ICON_iDigits_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 2 #Number of Clusters
    args.alpha = 1
    args.beta = 0.05
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0.5
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 

def init_ICON_DomainNet_args(args):

    args.IC = True  #
    args.CAST = True 

    args.d_threshold = True  #

    args.lr = 0.0028125

    args.opt_betas = [0.9, 0.999]
    args.batch_size = 24
    args.ema_decay = 0.9999
    args.k = 3 #Number of Clusters
    args.alpha = 1
    args.beta = 0.01
    args.gamma = 2

    args.adapt_blocks = [0, 1, 2, 3, 4] 
    args.thre = 0 
    args.reinit_optimizer = True
    args.clip_grad = 0.0
    args.num_freeze_epochs = 3  
    args.eval_only_emas = False 
    args.norm_cast = True 



