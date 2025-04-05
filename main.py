import sys
import argparse 
import datetime
import random
import numpy as np
import time
import torch
import importlib


from pathlib import Path

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import set_data_config, get_ood_dataset, get_dataset, find_tasks_with_unseen_classes, UnknownWrapper
import models #여기서 models.py의 @register_model이 실행되고, timm의 모델 레지스트리에 등록, create_model를 통해 custom vit가 호출됨
import utils
import os
from utils import seed_everything

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
warnings.filterwarnings("ignore","The given NumPy array is not writable, and PyTorch does not support non-writable tensors")


def main(args):
    args = set_data_config(args)
    device = torch.device(args.device)

    seed_everything(args.seed)
    
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    if args.ood_dataset:
        data_loader[-1]['ood'] = get_ood_dataset(args.ood_dataset, args)

    if args.develop_tasks: return

    try:
        engine_module = importlib.import_module(f"engines.{args.method}")
    except ImportError:
        raise ValueError(f"Unknown engine type: {args.method}")
    Engine = engine_module.Engine
    model = engine_module.load_model(args)
    model.to(args.device)
    engine = Engine(model=model, device=args.device, class_mask=class_mask, domain_list=domain_list, args=args)
    
    print(args)
    
    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                raise ValueError('No checkpoint found at:', checkpoint_path)
            
            # _ = engine.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args = args)

            print(f"{f'Task {task_id+1} OOD Evaluation':=^60}")
            ood_start = time.time()
            all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader[:task_id+1]])
            unseen_tasks = find_tasks_with_unseen_classes(task_id,class_mask)
            if unseen_tasks == []:
                print("No unseen tasks")
                continue
            print(unseen_tasks)
            ood_datasets =  torch.utils.data.ConcatDataset([UnknownWrapper(data_loader[i]['val'].dataset, args.num_classes) for i in unseen_tasks])
            engine.evaluate_ood(model, all_id_datasets, ood_datasets, device, args)
            ood_duration = time.time() - ood_start
            print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
        return
    
    if args.ood_eval:

        checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(args.num_tasks))
        if os.path.exists(checkpoint_path):
            print('Loading checkpoint from:', checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
        else:
            raise ValueError('No checkpoint found at:', checkpoint_path)
        
        if args.ood_dataset:
            print(f"{'OOD Evaluation':=^60}")
            ood_start = time.time()
            all_id_datasets = torch.utils.data.ConcatDataset([dl['val'].dataset for dl in data_loader])
            ood_loader = data_loader[-1]['ood']
            engine.evaluate_ood(model, all_id_datasets, ood_loader, device, args)
            ood_duration = time.time() - ood_start
            print(f"OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
        else:
            raise ValueError('No ood dataset')
        
        return
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    engine.train_and_evaluate(model,criterion, data_loader, optimizer, 
                       lr_scheduler, device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OOD-VIL')
    
    parser.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--method', default='ICON', type=str, help='Engine type to use (e.g., ICON, FT)')
    parser.add_argument('--model', default=None, type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')


    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    # Data parameters
    parser.add_argument('--data_path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='iDigits', type=str, help='dataset name')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--save', default='./save', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for data loading')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--IL_mode', type=str, default='cil', choices=['cil', 'dil', 'vil', 'joint'], help='Incremental Learning mode')

    # Misc (기타) parameters
    parser.add_argument('--print_freq', type=int, default=1000, help = 'The frequency of printing')
    parser.add_argument('--develop_tasks', action='store_true', default=False)
    parser.add_argument('--develop', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    # ood evaluation
    parser.add_argument('--ood_method', default='MSP', type=str, help='OOD detection method')
    parser.add_argument('--ood_dataset', default=None, type=str, help='OOD dataset name')
    parser.add_argument('--ood_threshold', default=0.5, type=float, help='OOD threshold')
    parser.add_argument('--ood_eval', action='store_true', help='Perform ood evaluation only')
    parser.add_argument('--normalize_ood_scores', default=True ,action='store_true', help='Normalize ood scores')
    args = parser.parse_args()

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    main(args)

    sys.exit(0)