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

# 추가: OOD 평가에 필요한 모듈들
from sklearn.metrics import roc_auc_score, confusion_matrix
from continual_datasets.dataset_utils import RandomSampleWrapper

def load_model(args):
    init_ICON_args(args)
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
        stat_matrix = np.zeros((3, args.num_tasks))
        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                       device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args)
            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[2, i] = test_stats['Loss']
            acc_matrix[i, task_id] = test_stats['Acc@1']
        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
        diagonal = np.diag(acc_matrix)
        A_avg = np.mean(acc_matrix[np.triu_indices(task_id + 1)])
        result_str = "[Average accuracy till task{}]\tA_last: {:.4f}\tA_avg: {:.4f}".format(task_id + 1, avg_stat[0], A_avg)
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
            result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
        print(result_str)
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
            if args.save and utils.is_main_process():
                Path(os.path.join(args.save, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                    'model': model.state_dict(),
                    'ema_model': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
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
        if args.ood_dataset:
            print(f"{'OOD Evaluation':=^60}")
            ood_start = time.time()
            # ID 데이터셋은 모든 태스크의 검증 데이터(ICON의 예측 로직 적용)를 합침
            all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader])
            # ood_loader는 마지막 태스크에 추가된 ood 데이터 로더 사용
            ood_loader = data_loader[-1]['ood']
            self.evaluate_ood(model, all_id_datasets, ood_loader, device, args)
            ood_duration = time.time() - ood_start
            print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")


    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args):
        model.eval()
        
        # OOD detection method 선택 (MSP, ENERGY, KL)
        ood_method = args.ood_method.upper()
        
        def MSP(logits):
            return F.softmax(logits, dim=1).max(dim=1)[0]

        def ENERGY(logits):
            return torch.logsumexp(logits, dim=1)
        
        def KL(logits):
            uniform = torch.ones_like(logits) / logits.shape[-1]
            return F.cross_entropy(logits, uniform, reduction='none')
        
        if ood_method == "MSP":
            infer_func = MSP
        elif ood_method == "ENERGY":
            infer_func = ENERGY
        elif ood_method == "KL":
            infer_func = KL
        else:
            raise ValueError(f"Unknown OOD detection method: {ood_method}")
        
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
        
        # raw anomaly score와 raw pred_class 저장 (thresholding은 나중에 수행)
        id_anomaly_scores_list = []
        id_pred_class_list = []
        id_true_labels = []
        
        ood_anomaly_scores_list = []
        ood_pred_class_list = []
        ood_true_labels = []
        
        # verbose 모드에서 logit 값을 출력하기 위한 리스트
        id_logits_list = []
        ood_logits_list = []
        
        with torch.no_grad():
            # ID 데이터 처리
            for inputs, targets in aligned_id_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs, _, _ = self.get_max_label_logits(outputs, list(range(self.num_classes)), slice=True)
                
                # 로그잇 저장 (verbose 모드)
                if args.verbose:
                    id_logits_list.append(outputs.detach().cpu())
                
                # raw anomaly score 저장
                scores = infer_func(outputs)
                id_anomaly_scores_list.append(scores)
                
                # raw softmax 예측 (argmax) 저장 (thresholding은 나중에)
                softmax_outputs = F.softmax(outputs, dim=1)
                pred_class = torch.max(softmax_outputs, dim=1)[1]
                id_pred_class_list.append(pred_class)
                id_true_labels.append(targets.to(device))
            
            # OOD 데이터 처리
            for inputs, _ in aligned_ood_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs, _, _ = self.get_max_label_logits(outputs, list(range(self.num_classes)), slice=True)
                
                # 로그잇 저장 (verbose 모드)
                if args.verbose:
                    ood_logits_list.append(outputs.detach().cpu())
                
                scores = infer_func(outputs)
                ood_anomaly_scores_list.append(scores)
                
                softmax_outputs = F.softmax(outputs, dim=1)
                pred_class = torch.max(softmax_outputs, dim=1)[1]
                ood_pred_class_list.append(pred_class)
                # OOD의 true label은 unknown (args.num_classes)
                true_labels = torch.full((outputs.size(0),), fill_value=args.num_classes, dtype=torch.long).to(device)
                ood_true_labels.append(true_labels)
        
        # 텐서로 합치기
        id_anomaly_scores = torch.cat(id_anomaly_scores_list, dim=0)
        ood_anomaly_scores = torch.cat(ood_anomaly_scores_list, dim=0)
        # 전체 anomaly score에서 global min, max로 정규화
        all_scores_tensor = torch.cat([id_anomaly_scores, ood_anomaly_scores], dim=0)
        if args.normalize_ood_scores:
            min_score = all_scores_tensor.min()
            max_score = all_scores_tensor.max()
            id_anomaly_scores_norm = (id_anomaly_scores - min_score) / (max_score - min_score)
            ood_anomaly_scores_norm = (ood_anomaly_scores - min_score) / (max_score - min_score)
        else:
            id_anomaly_scores_norm = id_anomaly_scores
            ood_anomaly_scores_norm = ood_anomaly_scores
        
        # verbose: anomaly score 히스토그램 저장
        if args.verbose:
            from utils import save_anomaly_histogram
            save_anomaly_histogram(id_anomaly_scores_norm.cpu().numpy(), ood_anomaly_scores_norm.cpu().numpy(), args)
        
        # thresholding: 정규화된 anomaly score를 기준으로 판별
        id_pred_class = torch.cat(id_pred_class_list, dim=0)
        ood_pred_class = torch.cat(ood_pred_class_list, dim=0)
        
        id_preds = torch.where(id_anomaly_scores_norm < args.ood_threshold,
                            torch.full_like(id_pred_class, args.num_classes),
                            id_pred_class)
        ood_preds = torch.where(ood_anomaly_scores_norm < args.ood_threshold,
                                torch.full_like(ood_pred_class, args.num_classes),
                                ood_pred_class)
        
        id_true = torch.cat(id_true_labels, dim=0)
        ood_true = torch.cat(ood_true_labels, dim=0)
        
        # verbose: confusion matrix 및 로그잇, 레이블 샘플 출력
        if args.verbose:
            # ID와 OOD의 예측 결과 및 정답 결합
            all_preds = torch.cat([id_preds, ood_preds], dim=0)
            all_trues = torch.cat([id_true, ood_true], dim=0)
            from sklearn.metrics import confusion_matrix
            conf_mat = confusion_matrix(all_trues.cpu().numpy(), all_preds.cpu().numpy())
            print("Confusion Matrix:")
            print(conf_mat)
            
            # 첫 번째 배치의 샘플 로그잇과 레이블 출력 (샘플 수: 5)
            if id_logits_list:
                print("Sample ID logits (first 5 samples from first batch):")
                print(id_logits_list[0][:5])
            if ood_logits_list:
                print("Sample OOD logits (first 5 samples from first batch):")
                print(ood_logits_list[0][:5])
            if id_true_labels:
                print("Sample ID labels (first batch):")
                print(id_true_labels[0])
            if ood_true_labels:
                print("Sample OOD labels (first batch):")
                print(ood_true_labels[0])
        
        acc_id = (id_preds == id_true).float().mean().item()
        acc_ood = (ood_preds == ood_true).float().mean().item()
        h_score = 2 * acc_id * acc_ood / (acc_id + acc_ood) if (acc_id + acc_ood) > 0 else 0.0
        
        # labels: 0 = OOD, 1 = ID
        # scores: it is anomality score (the higher the score, the more anomalous)
        binary_labels = np.concatenate([np.ones(id_anomaly_scores_norm.shape[0]),
                                        np.zeros(ood_anomaly_scores_norm.shape[0])])
        # 전체 normalized anomaly score 결합 (numpy array)
        all_scores = torch.cat([id_anomaly_scores_norm, ood_anomaly_scores_norm], dim=0).cpu().numpy()
        
        # ROC 및 기타 지표 계산
        from sklearn import metrics
        fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
        auroc = metrics.auc(fpr, tpr)
        idx_tpr95 = np.abs(tpr - 0.95).argmin()
        fpr_at_tpr95 = fpr[idx_tpr95]
        dtacc = 0.5 * (tpr + 1 - fpr).max()
        auprc_in = metrics.average_precision_score(y_true=binary_labels, y_score=all_scores)
        auprc_out = metrics.average_precision_score(y_true=binary_labels, y_score=-all_scores, pos_label=0)
        
        print(f"[{ood_method}]: evaluating metrics...")
        print(f"ID classification accuracy (A_id): {acc_id * 100:.2f}%")
        print(f"OOD detection accuracy (A_ood): {acc_ood * 100:.2f}%")
        print(f"Harmonic mean (H_score): {h_score * 100:.2f}%")
        print(f"AUROC: {auroc * 100:.2f}%, FPR@TPR95: {fpr_at_tpr95 * 100:.2f}%, dtacc: {dtacc * 100:.2f}%")
        print(f"AUPRC in: {auprc_in * 100:.2f}%, AUPRC out: {auprc_out * 100:.2f}%")
        
        return all_scores, binary_labels, acc_id






def init_ICON_args(args):
    args.IC = True  
    args.d_threshold = True  
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
