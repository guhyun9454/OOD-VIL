# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

def save_logits_statistics(id_logits, ood_logits, args, task_id):
    """
    logits의 통계값을 계산하고 시각화하여 저장합니다. 통계값은 출력만 합니다.
    
    Args:
        id_logits: ID 데이터의 logits
        ood_logits: OOD 데이터의 logits
        args: 설정 인자
        task_id: 현재 태스크 ID
    """
    if not os.path.exists(os.path.join(args.save, 'logits_stats')):
        os.makedirs(os.path.join(args.save, 'logits_stats'))
    
    # ID 데이터의 logits 통계
    id_mean = torch.mean(id_logits, dim=0).cpu().numpy()
    id_max = torch.max(id_logits, dim=0)[0].cpu().numpy()
    id_min = torch.min(id_logits, dim=0)[0].cpu().numpy()
    id_std = torch.std(id_logits, dim=0).cpu().numpy()
    
    # OOD 데이터의 logits 통계
    ood_mean = torch.mean(ood_logits, dim=0).cpu().numpy()
    ood_max = torch.max(ood_logits, dim=0)[0].cpu().numpy()
    ood_min = torch.min(ood_logits, dim=0)[0].cpu().numpy()
    ood_std = torch.std(ood_logits, dim=0).cpu().numpy()
    
    # 클래스별 logits 평균 시각화
    plt.figure(figsize=(12, 6))
    x = np.arange(len(id_mean))
    plt.bar(x - 0.2, id_mean, width=0.4, label='ID Mean', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_mean, width=0.4, label='OOD Mean', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Mean Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Mean Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_mean_logits.png'))
    plt.close()
    
    # 클래스별 logits 최대값 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, id_max, width=0.4, label='ID Max', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_max, width=0.4, label='OOD Max', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Max Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Max Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_max_logits.png'))
    plt.close()
    
    # 클래스별 logits 표준편차 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, id_std, width=0.4, label='ID Std', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_std, width=0.4, label='OOD Std', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Std Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Logit Standard Deviations')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_std_logits.png'))
    plt.close()
    
    # Logits 분포 히스토그램 (전체 logits의 분포)
    plt.figure(figsize=(12, 6))
    plt.hist(id_logits.flatten().cpu().numpy(), bins=50, alpha=0.7, label='ID Logits', color='blue')
    plt.hist(ood_logits.flatten().cpu().numpy(), bins=50, alpha=0.7, label='OOD Logits', color='red')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title(f'Task {task_id+1}: Distribution of Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_logits_distribution.png'))
    plt.close()
    
    # 통계값 출력 (txt 파일 저장 대신)
    print(f"\nTask {task_id+1} Logits Statistics")
    print("="*50)
    print("ID Data Statistics:")
    print(f"Mean: {np.mean(id_mean):.4f}")
    print(f"Max: {np.max(id_max):.4f}")
    print(f"Min: {np.min(id_min):.4f}")
    print(f"Std: {np.mean(id_std):.4f}\n")
    
    print("OOD Data Statistics:")
    print(f"Mean: {np.mean(ood_mean):.4f}")
    print(f"Max: {np.max(ood_max):.4f}")
    print(f"Min: {np.min(ood_min):.4f}")
    print(f"Std: {np.mean(ood_std):.4f}")
    print("="*50)

    return os.path.join(args.save, 'logits_stats', f'task{task_id+1}_logits_distribution.png')

def save_confusion_matrix_plot(confusion_matrix, labels, args, task_id=None):
    # Task 별 폴더 생성
    task_folder = f"task{task_id+1}" if task_id is not None else "latest"
    task_path = os.path.join(args.save, task_folder)
    os.makedirs(task_path, exist_ok=True)
    
    # 파일명 생성 (task 정보 포함)
    file_name = "confusion_matrix"
    if task_id is not None:
        file_name += f"_task{task_id+1}"
    
    save_path = os.path.join(task_path, f"{file_name}.png")
    
    modified_labels = labels.copy()
    modified_labels[-1] = 'ood'
    plt.figure(figsize=(16,12))
    sns.heatmap(confusion_matrix,
                annot=False,
                fmt='d',
                cmap='Blues',
                xticklabels=modified_labels,
                yticklabels=modified_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    title = "Confusion Matrix"
    if task_id is not None:
        title += f" - Task {task_id+1}"
    
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_anomaly_histogram(id_scores, ood_scores, args, suffix='', task_id=None):
    plt.figure(figsize=(10, 6))
    
    # 폴더 생성
    save_dir = os.path.join(args.save, 'anomaly_histograms')
    os.makedirs(save_dir, exist_ok=True)
    
    # ID 및 OOD 점수 히스토그램 그리기
    bins = np.linspace(min(np.min(id_scores), np.min(ood_scores)), 
                      max(np.max(id_scores), np.max(ood_scores)), 100)
    
    plt.hist(id_scores, bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(ood_scores, bins=bins, alpha=0.5, label='OOD', density=True)
    
    plt.title(f'Anomaly Score Distribution ({suffix.upper()})')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    
    task_str = f'_task{task_id+1}' if task_id is not None else ''
    save_path = os.path.join(save_dir, f'anomaly_hist_{suffix}{task_str}.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def save_accuracy_heatmap(acc_matrix, task_id, args):
    # 폴더 생성
    save_dir = os.path.join(args.save, 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(acc_matrix, annot=True, cmap='YlGnBu', ax=ax, vmin=0, vmax=100)
    
    ax.set_title(f'Accuracy Heatmap (Task {task_id+1})')
    ax.set_xlabel('After learning task')
    ax.set_ylabel('Tested on task')
    
    # 저장 경로 설정
    save_path = os.path.join(save_dir, f'heatmap_task{task_id+1}.png')
    plt.savefig(save_path)
    plt.close()
    
    return save_path

import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save({'state_dict_ema':checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
