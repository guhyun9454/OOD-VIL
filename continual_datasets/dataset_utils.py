# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# https://github.com/pytorch/vision/blob/8635be94d1216f10fb8302da89233bd86445e449/torchvision/datasets/utils.py

import numpy as np
import torch
import torch 
from torch.utils.model_zoo import tqdm
from torchvision import transforms
from continual_datasets.base_datasets import *

def get_dataset(dataset, transform_train, transform_val, mode, args,is_ood=False):
    if dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)

    elif dataset == 'MNISTM':
        dataset_train = MNISTM(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNISTM(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'SynDigit':
        dataset_train = SynDigit(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = SynDigit(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'EMNIST':
        dataset_train = EMNIST_RGB(args.data_path, train=True, download=True, transform=transform_train, num_random_classes=10, split='letters')
        dataset_val = EMNIST_RGB(args.data_path, train=False, download=True, transform=transform_val, num_random_classes=10, split='letters')

    elif dataset == 'CORe50':
        dataset_train = CORe50(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CORe50(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'DomainNet':
        dataset_train = DomainNet(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = DomainNet(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data

    elif dataset == 'CLEAR':
        dataset_train = CLEAR(args.data_path, train=True, download=True, transform=transform_train, mode=mode).data
        dataset_val = CLEAR(args.data_path, train=False, download=True, transform=transform_val, mode=mode).data
        
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    if args.verbose:
        divider = "=" * 60
        print(divider)
        print(f"Dataset: {dataset}")
        # Train dataset 정보 출력
        if isinstance(dataset_train, list):
            total_train = sum(len(ds) for ds in dataset_train)
            print(f"Train dataset total size: {total_train}")
            for i, ds in enumerate(dataset_train):
                try:
                    classes = ds.classes
                    print(f"  Sub-dataset {i}: size {len(ds)}, {len(classes)} classes, classes: {classes}")
                except AttributeError:
                    print(f"  Sub-dataset {i}: size {len(ds)}")
        else:
            print(f"Train dataset size: {len(dataset_train)}")
            try:
                print(f"Number of classes: {len(dataset_train.classes)}")
                print(f"Classes: {dataset_train.classes}")
            except AttributeError:
                pass

        # Validation dataset 정보 출력
        if isinstance(dataset_val, list):
            total_val = sum(len(ds) for ds in dataset_val)
            print(f"Validation dataset total size: {total_val}")
            for i, ds in enumerate(dataset_val):
                try:
                    classes = ds.classes
                    print(f"  Sub-dataset {i}: size {len(ds)}, {len(classes)} classes, classes: {classes}")
                except AttributeError:
                    print(f"  Sub-dataset {i}: size {len(ds)}")
        else:
            print(f"Validation dataset size: {len(dataset_val)}")
            try:
                print(f"Number of classes: {len(dataset_val.classes)}")
                print(f"Classes: {dataset_val.classes}")
            except AttributeError:
                pass
        print(divider)
    
    return dataset_train, dataset_val

def get_ood_dataset(dataset_name, args):
    if args.verbose:
        print(f"Loading OOD dataset: {dataset_name}")
    dataset = get_dataset(dataset_name, transform_train=build_transform(True,args), transform_val=build_transform(False,args), mode='joint', args=args)[0]
    ood_dataset = UnknownWrapper(dataset, args.class_num)
    return ood_dataset

def set_data_config(args):
    if args.dataset == "iDigits":
        args.class_num = 10
        args.domain_num = 4
        args.id_datasets = ['MNIST', 'SVHN', 'MNISTM', 'SynDigit']
    elif args.dataset == "DomainNet":
        args.class_num = 345
        args.domain_num = 6
    elif args.dataset == "CORe50":
        args.class_num = 50
        args.domain_num = 8
    elif args.dataset == "CLEAR":
        args.class_num = 100
        args.domain_num = 5
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return args

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

class UnknownWrapper(torch.utils.data.Dataset):
    """
    원본 데이터셋의 라벨을 모두 unknown_label(= num_known)로 변경합니다.
    """
    def __init__(self, dataset, unknown_label):
        self.dataset = dataset
        self.unknown_label = unknown_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        
        return x, self.unknown_label

class RandomSampleWrapper(torch.utils.data.Dataset):
    """
    주어진 데이터셋에서 num_samples만큼 랜덤으로 샘플링하여 반환합니다.
    """
    def __init__(self, dataset, num_samples, seed):
        self.dataset = dataset
        self.num_samples = num_samples
        np.random.seed(seed)
        # replacement 없이 num_samples 개의 인덱스 선택
        self.indices = np.random.choice(len(dataset), size=num_samples, replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


