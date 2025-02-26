import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None
    domain_list = None
    
    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.task_inc:
        mode = 'til'
    elif args.domain_inc:
        mode = 'dil'
    elif args.versatile_inc:
        mode = 'vil'
    elif args.joint_train:
        mode = 'joint'
    else:
        mode = 'cil'

    if mode in ['til', 'cil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )

                splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
                mask.append(class_mask)

                for i in range(len(splited_dataset)):
                    train.append(splited_dataset[i][0])
                    val.append(splited_dataset[i][1])
            
            #mask = [[[0,1], [2,3], [4,5], [6,7], [8,9]] , [[0,1], [2,3], [4,5], [6,7], [8,9]], ...] 4개의 데이터셋에 대해 각각의 태스크별 클래스 마스크
            #train = [train, train, ... ] 20개
            #val = [val, val, ... ] 20개
                    
            splited_dataset = list()
            for i in range(args.num_tasks): #5  
                t = [train[i+args.num_tasks*j] for j in range(len(dataset_list))]
                v = [val[i+args.num_tasks*j] for j in range(len(dataset_list))]
                splited_dataset.append((torch.utils.data.ConcatDataset(t), torch.utils.data.ConcatDataset(v)))

            #splited_dataset = [(train, val), (train, val), (train, val), (train, val), (train, val)]
            #class_mask = [[0,1], [2,3], [4,5], [6,7], [8,9]]
            args.nb_classes = len(splited_dataset[0][1].datasets[0].dataset.classes)
            class_mask = np.unique(np.array(mask), axis=0).tolist()[0] 
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
            args.nb_classes = len(dataset_val.classes)

    elif mode in ['dil', 'vil']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            splited_dataset = list()

            for i in range(len(dataset_list)):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset_list[i],
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                splited_dataset.append((dataset_train, dataset_val))
            #splited_dataset = [(train, val), (train, val), (train, val), (train, val)] 각 d0,d1,d2,d3
            args.nb_classes = len(dataset_val.classes)
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            if args.dataset in ['CORe50']:
                splited_dataset = [(dataset_train[i], dataset_val) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val.classes)
            else:
                splited_dataset = [(dataset_train[i], dataset_val[i]) for i in range(len(dataset_train))]
                args.nb_classes = len(dataset_val[0].classes)
    
    elif mode in ['joint']:
        if 'iDigits' in args.dataset:
            dataset_list = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
            train, val = list(), list()
            mask = list()
            for i, dataset in enumerate(dataset_list):
                dataset_train, dataset_val = get_dataset(
                    dataset=dataset,
                    transform_train=transform_train,
                    transform_val=transform_val,
                    mode=mode,
                    args=args,
                )
                train.append(dataset_train)
                val.append(dataset_val)
                args.nb_classes = len(dataset_val.classes)

            dataset_train = torch.utils.data.ConcatDataset(train)
            dataset_val = torch.utils.data.ConcatDataset(val)
            splited_dataset = [(dataset_train, dataset_val)]

            class_mask = None
        
        else:
            dataset_train, dataset_val = get_dataset(
                dataset=args.dataset,
                transform_train=transform_train,
                transform_val=transform_val,
                mode=mode,
                args=args,
            )

            splited_dataset = [(dataset_train, dataset_val)]

            args.nb_classes = len(dataset_val.classes)
            class_mask = None
            
    else:
        raise ValueError(f'Invalid mode: {mode}')
                

    if args.versatile_inc:
        splited_dataset, class_mask, domain_list, args = build_vil_scenario(splited_dataset, args)
        for c, d in zip(class_mask, domain_list):
            print(c, d)
    for i in range(len(splited_dataset)):
        dataset_train, dataset_val = splited_dataset[i]

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val})

    return dataloader, class_mask, domain_list

def get_dataset(dataset, transform_train, transform_val, mode, args,):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes) # [0,1,2,3,4,5,6,7,8,9] -> 10
    assert nb_classes % args.num_tasks == 0 # 10 % 5 = 0
    classes_per_task = nb_classes // args.num_tasks # 10 // 5 = 2

    labels = [i for i in range(nb_classes)] # [0,1,2,3,4,5,6,7,8,9]
    
    split_datasets = list() # [[train, val], [train, val], [train, val], [train, val], [train, val]]
    mask = list() # [[0,1], [2,3], [4,5], [6,7], [8,9]]

    if args.shuffle:
        random.shuffle(labels)

    for _ in range(args.num_tasks): # 5
        train_split_indices = list()  # [0,1]
        test_split_indices = list() # [2,3,4,5,6,7,8,9]
        
        scope = labels[:classes_per_task] # [0,1]
        labels = labels[classes_per_task:] # [2,3,4,5,6,7,8,9]

        mask.append(scope)

        #학습 데이터셋과 검증 데이터셋의 각 샘플에 대해, 해당 샘플의 레이블이 현재 scope에 속하면 그 인덱스를 선택
        for k in range(len(dataset_train.targets)): 
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)

        #선택된 인덱스들을 이용해 torch.utils.data.Subset을 생성하고, 이 서브셋을 현재 태스크의 학습/검증 데이터로 사용
        subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])
    
    return split_datasets, mask # [[train, val], [train, val], ...], [[0,1], [2,3], [4,5], [6,7], [8,9]]

def build_vil_scenario(splited_dataset, args):
    datasets = list()
    class_mask = list()
    domain_list = list()

    #splited_dataset = [(train, val), (train, val), (train, val), (train, val)] 각 d0,d1,d2,d3
    for i in range(len(splited_dataset)):
        dataset, mask = split_single_dataset(splited_dataset[i][0], splited_dataset[i][1], args)
        datasets.append(dataset)
        class_mask.append(mask)
        for _ in range(len(dataset)):
            domain_list.append(f'D{i}')

    splited_dataset = sum(datasets, [])
    class_mask = sum(class_mask, [])

    args.num_tasks = len(splited_dataset)

    zipped = list(zip(splited_dataset, class_mask, domain_list))
    random.shuffle(zipped)
    splited_dataset, class_mask, domain_list = zip(*zipped)

    return splited_dataset, class_mask, domain_list, args

def build_transform(is_train, args):
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    return transform