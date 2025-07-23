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

def update_ood_hyperparams(args):

    from OODdetectors import ood_adapter as _oa


    # ENERGY
    _oa._DEFAULT_PARAMS.setdefault("ENERGY", {})["temperature"] = args.energy_temperature

    # GEN (공통: GEN, PRO_GEN)
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["M"] = args.gen_M

    # PRO_GEN
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["M"] = args.gen_M
    _oa._DEFAULT_PARAMS["PRO_GEN"]["noise_level"] = args.pro_gen_noise_level
    _oa._DEFAULT_PARAMS["PRO_GEN"]["gd_steps"] = args.pro_gen_gd_steps

    # PRO_MSP
    _oa._DEFAULT_PARAMS.setdefault("PRO_MSP", {})["temperature"] = args.pro_msp_temperature
    _oa._DEFAULT_PARAMS["PRO_MSP"]["noise_level"] = args.pro_msp_noise_level
    _oa._DEFAULT_PARAMS["PRO_MSP"]["gd_steps"] = args.pro_msp_gd_steps

    # PRO_MSP_T
    _oa._DEFAULT_PARAMS.setdefault("PRO_MSP_T", {})["temperature"] = args.pro_msp_t_temperature
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["noise_level"] = args.pro_msp_t_noise_level
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["gd_steps"] = args.pro_msp_t_gd_steps

    # PRO_ENT
    _oa._DEFAULT_PARAMS.setdefault("PRO_ENT", {})["noise_level"] = args.pro_ent_noise_level
    _oa._DEFAULT_PARAMS["PRO_ENT"]["gd_steps"] = args.pro_ent_gd_steps

# ====== 샘플 시각화 유틸리티 ======

def _save_grid_chunk(chunk_imgs, chunk_titles, save_path, cols: int, dpi: int = 100):
    """내부 함수: 주어진 이미지/타이틀 목록을 하나의 그림으로 저장"""
    import math
    import matplotlib.pyplot as plt

    rows = math.ceil(len(chunk_imgs) / cols)
    # fig 크기는 rows, cols 에 비례하되 너무 커지지 않도록 1.8 inch 스케일 사용
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))

    # axes 가 1D 로 반환될 수 있는 경우(rows==1 또는 cols==1) 대비
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            axes = np.expand_dims(axes, 0 if rows == 1 else 1)
    else:  # 단일 Axes 객체
        axes = np.array([[axes]])

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx < len(chunk_imgs):
            img = chunk_imgs[idx]
            try:
                # Torch Tensor 처리
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu()
                    if img.ndim == 3:
                        img = img.permute(1, 2, 0)  # CHW -> HWC
                    img = img.numpy()
                    # 범위 조정 (0~1 or 0~255)
                    if img.max() > 1.0:
                        img = img / 255.0
                # numpy ndarray 처리 (Tensor가 아닌 경우)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[0] in (1, 3):
                        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
                        if img.shape[2] == 1:  # Grayscale 채널 제거
                            img = img.squeeze(2)
                    if img.max() > 1.0:
                        img = img / 255.0
                ax.imshow(img)
            except Exception as e:
                # 마지막 fallback: numpy 변환 후 시도
                try:
                    ax.imshow(np.array(img))
                except Exception:
                    print(f"[Warning] 이미지 표시 실패 (index {idx}): {e}")
            ax.set_title(chunk_titles[idx], fontsize=6)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_grid(images, titles, save_path, cols: int = 10, dpi: int = 100):
    """이미지들을 그리드 형태로 저장하되, 너무 큰 그림은 자동 분할합니다.

    Args:
        images (List[PIL.Image]): 이미지 객체 리스트
        titles (List[str]): 각 이미지에 대한 제목(캡션)
        save_path (str): 저장 경로 (확장자 포함). 여러 파트로 분할 시 _part{n}.png 추가됨.
        cols (int): 한 행에 배치할 열 수
        dpi (int): 저장 DPI (기본 100)
    """
    import math, os

    assert len(images) == len(titles), "images와 titles의 길이가 다릅니다."

    max_pixels = 65000  # matplotlib 한 변 제한 (2**16)
    max_rows = max_pixels // (2 * dpi)  # 각 행 높이를 2 inch 로 가정

    rows_total = math.ceil(len(images) / cols)

    # 분할이 필요한지 확인
    if rows_total <= max_rows:
        _save_grid_chunk(images, titles, save_path, cols, dpi)
        return

    # 분할 저장
    chunk_size = cols * max_rows  # 한 파트당 최대 이미지 수
    num_parts = math.ceil(len(images) / chunk_size)

    base, ext = os.path.splitext(save_path)
    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min((part_idx + 1) * chunk_size, len(images))
        chunk_imgs = images[start:end]
        chunk_titles = titles[start:end]
        part_save_path = f"{base}_part{part_idx+1}{ext}"
        _save_grid_chunk(chunk_imgs, chunk_titles, part_save_path, cols, dpi)
        print(f"Saved grid part {part_idx+1}/{num_parts} -> {part_save_path}")


def visualize_samples(args, data_loader, class_mask, domain_list):
    """ID / OOD 샘플을 모아 그리드 이미지로 저장합니다.

    - ID: 도메인-클래스 조합당 하나씩
    - OOD: 클래스당 하나씩
    """
    from pathlib import Path
    from PIL import Image
    from collections import defaultdict
    import torch

    save_dir = os.path.join(args.save, "sample_visualization")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    domain_to_classimages: dict[str, dict[str, "Image"]] = defaultdict(dict)

    for t_idx, task in enumerate(data_loader[: args.num_tasks]):
        domain = domain_list[t_idx] if domain_list is not None else f"D{t_idx}"
        classes = class_mask[t_idx] if class_mask is not None else []
        dataset = task["val"].dataset

        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            indices = dataset.indices
        else:
            base_dataset = dataset
            indices = range(len(dataset))

        for cls_idx in classes:
            cls_name = (
                base_dataset.classes[cls_idx] if hasattr(base_dataset, "classes") else str(cls_idx)
            )
            if cls_name in domain_to_classimages[domain]:
                continue

            found_path = None
            found_img = None  # 이미지 객체 직접 저장
            if hasattr(base_dataset, "imgs"):
                # ImageFolder 계열: 파일 경로 리스트 존재
                for idx in indices:
                    path, label = base_dataset.imgs[idx]
                    if label == cls_idx:
                        found_path = path
                        try:
                            from PIL import Image as _PILImage
                            found_img = _PILImage.open(found_path).convert("RGB")
                        except Exception:
                            found_img = None
                        break
            else:
                # .imgs 가 없으면 __getitem__을 통해 직접 로드
                for idx in indices:
                    try:
                        img, label = base_dataset[idx]
                    except Exception:
                        continue  # 특정 샘플 로드 실패 시 건너뜀
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    if label == cls_idx:
                        # img 가 Tensor / PIL / ndarray 등 다양한 타입일 수 있으므로 그대로 저장
                        found_img = img
                        break

            # 이미지 저장
            if found_img is not None:
                domain_to_classimages[domain][cls_name] = found_img

    # 도메인별로 개별 저장
    for dom in sorted(domain_to_classimages.keys()):
        imgs = list(domain_to_classimages[dom].values())
        titles = list(domain_to_classimages[dom].keys())
        if imgs:
            file_name = f"id_{dom}.png"
            plot_grid(imgs, titles, os.path.join(save_dir, file_name))
            print(f"ID 도메인 {dom} 샘플 그리드를 {os.path.join(save_dir, file_name)} 에 저장했습니다.")

    # -------------------- OOD 데이터 --------------------
    if "ood" in data_loader[-1]:
        ood_dataset_wrapper = data_loader[-1]["ood"]
        base_dataset = getattr(ood_dataset_wrapper, "dataset", ood_dataset_wrapper)

        class_to_img = {}

        # 1) ImageFolder 류 (.imgs 보유) 우선 처리
        if hasattr(base_dataset, "imgs") and len(base_dataset.imgs) > 0:
            for path, label in base_dataset.imgs:
                if label not in class_to_img:
                    class_to_img[label] = Image.open(path).convert("RGB")
                if hasattr(base_dataset, "classes") and len(class_to_img) == len(base_dataset.classes):
                    break
        # 2) 그렇지 않으면 __getitem__으로 순회하며 수집
        if not class_to_img:
            for idx in range(len(base_dataset)):
                try:
                    img, label = base_dataset[idx]
                except Exception as e:
                    # 일부 샘플에서 오류 발생 시 건너뜀
                    print(f"[Warning] OOD 샘플 로드 실패 idx={idx}: {e}")
                    continue
                # label tensor -> int
                if isinstance(label, torch.Tensor):
                    label = label.item()
                if label not in class_to_img:
                    # img 가 Tensor 면 numpy 로 변환 필요 없이 imshow 사용 가능
                    # 단, PIL type 아닌 경우 imshow를 위해 shape 변환 필요 없음 (matplotlib 지원)
                    class_to_img[label] = img
                # 클래스 수 파악
                if hasattr(base_dataset, "classes") and len(class_to_img) == len(base_dataset.classes):
                    break

        if class_to_img:
            ood_images, ood_titles = [], []
            for label_idx, img in class_to_img.items():
                class_name = (
                    base_dataset.classes[label_idx]
                    if hasattr(base_dataset, "classes")
                    else str(label_idx)
                )
                ood_images.append(img)
                ood_titles.append(class_name)

            plot_grid(ood_images, ood_titles, os.path.join(save_dir, "ood_samples.png"))
            print(f"OOD 샘플 그리드를 {os.path.join(save_dir, 'ood_samples.png')} 에 저장했습니다.")
        else:
            print("[Warning] OOD 샘플을 찾지 못했습니다 – 시각화 생략.")
