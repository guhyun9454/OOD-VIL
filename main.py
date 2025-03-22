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
from continual_datasets.dataset_utils import set_data_config, get_ood_dataset, get_dataset
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
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):
                print('Loading checkpoint from:', checkpoint_path)
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
            else:
                raise ValueError('No checkpoint found at:', checkpoint_path)
            
            _ = engine.evaluate_till_now(model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)

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
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: (0.9, 0.999), use opt default)')
    parser.add_argument('--clip-grad', type=float, default=0.0, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')
    parser.add_argument('--reinit_optimizer', type=bool, default=True, help='reinit optimizer (default: True)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.0028125, metavar='LR', help='learning rate (default: 0.03)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')
    parser.add_argument('--unscale_lr', type=bool, default=True, help='scaling lr by batch size (default: True)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    # Data parameters
    parser.add_argument('--data-path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='iDigits', type=str, help='dataset name')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--IL_mode', type=str, default='cil', choices=['cil', 'dil', 'vil', 'ood_vil', 'joint'], help='Incremental Learning mode')


    # Prompt parameters
    parser.add_argument('--adapt_blocks', default=[0, 1, 2, 3, 4])
    parser.add_argument('--ema_decay', default=0.9999)
    parser.add_argument('--num_freeze_epochs', type=int,default=3)
    parser.add_argument('--eval_only_emas', default=False)

    # Misc (기타) parameters
    parser.add_argument('--print_freq', type=int, default=1000, help = 'The frequency of printing')
    parser.add_argument('--develop', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)


    # ood evaluation
    parser.add_argument('--ood_dataset', default=None, type=str, help='OOD dataset name')
    parser.add_argument('--ood_threshold', default=0.5, type=float, help='OOD threshold')
    
     #! IC
    parser.add_argument('--IC', action='store_true', default=False, help='if using incremental classifier')
    parser.add_argument('--d_threshold', action='store_true', default=False, help='if using dynamic thresholding in IC')
    parser.add_argument('--gamma',default=10.0, type=float, help='coefficient in dynamic thresholding')
    parser.add_argument('--thre',default=0, type=float, help='value of static threshold if not using dynamic thresholding')
    parser.add_argument('--alpha',default=1.0, type=float, help='coefficient of knowledge distillation in IC loss')

    #! CAST
    parser.add_argument('--beta',default=0.01, type=float, help='coefficient of cast loss')
    parser.add_argument('--k', default=2, type=int, help='the number of clusters in shift pool')
    parser.add_argument('--CAST', action='store_true', default=False, help='if using CAST loss')
    parser.add_argument('--norm_cast', action='store_true', default=False, help='if using normalization in cast')
    
   
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    main(args)

    sys.exit(0)