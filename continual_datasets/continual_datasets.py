import os
import os.path

import pathlib
from pathlib import Path

from typing import Any, Tuple

import glob
from shutil import move, rmtree

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive, extract_archive

import PIL
from PIL import Image

import tqdm
import zipfile
import tarfile

from .dataset_utils import read_image_file, read_label_file

import string
import shutil
from typing import Any, Tuple, Union

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        self.classes = [i for i in range(10)]

    def _check_legacy_exist(self):
        processed_folder_exists = os.path.exists(self.processed_folder)
        if not processed_folder_exists:
            return False

        return all(
            check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
        )

    def _load_legacy_data(self):
        # This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
        # directly.
        data_file = self.training_file if self.train else self.test_file
        return torch.load(os.path.join(self.processed_folder, data_file))

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTM(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'MNIST-M')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SynDigit(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'SynDigit')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

#torchvision 0.21 version
class EMNIST(datasets.MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``EMNIST/raw/train-images-idx3-ubyte``
            and  ``EMNIST/raw/t10k-images-idx3-ubyte`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        "byclass": sorted(list(_all_classes)),
        "bymerge": sorted(list(_all_classes - _merged_classes)),
        "balanced": sorted(list(_all_classes - _merged_classes)),
        "letters": ["N/A"] + list(string.ascii_lowercase),
        "digits": list(string.digits),
        "mnist": list(string.digits),
    }

    def __init__(self, root: Union[str, Path], split: str, **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super().__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return f"training_{split}.pt"

    @staticmethod
    def _test_file(split) -> str:
        return f"test_{split}.pt"

    @property
    def _file_prefix(self) -> str:
        return f"emnist-{self.split}-{'train' if self.train else 'test'}"

    @property
    def images_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-images-idx3-ubyte")

    @property
    def labels_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-labels-idx1-ubyte")

    def _load_data(self):
        return read_image_file(self.images_file), read_label_file(self.labels_file)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        download_and_extract_archive(self.url, download_root=self.raw_folder, md5=self.md5)
        gzip_folder = os.path.join(self.raw_folder, "gzip")
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith(".gz"):
                extract_archive(os.path.join(gzip_folder, gzip_file), self.raw_folder)
        shutil.rmtree(gzip_folder)

    
class EMNIST_RGB(EMNIST):
    def __init__(self, root, split='letters', train=True, transform=None, target_transform=None, download=False,
                 random_seed=42, num_random_classes=10):
        super(EMNIST_RGB, self).__init__(root, split=split, train=train, transform=transform,
                                         target_transform=target_transform, download=download)
        # letters split인 경우 "N/A" (target 0)를 제외한 알파벳(a~z)이 target 1~26로 매핑됨.
        # num_random_classes가 지정되면, valid target (1~26) 중에서 랜덤으로 선택하여 필터링합니다.
        if split == 'letters' and num_random_classes is not None:
            valid_letters = list(string.ascii_lowercase)  # ['a', 'b', ... 'z']
            valid_targets = np.arange(1, 27)  # 1 ~ 26
            if random_seed is not None:
                np.random.seed(random_seed)
            # 랜덤으로 num_random_classes 개를 샘플링 (중복 없이)
            sampled = np.random.choice(valid_targets, size=num_random_classes, replace=False)
            sampled = np.sort(sampled)  # 정렬해서 매핑 순서를 일정하게 유지
            # self.targets는 torch.Tensor임 (타입: torch.long)
            target_np = self.targets.numpy()
            valid_mask = np.isin(target_np, sampled)
            self.data = self.data[valid_mask]
            self.targets = self.targets[valid_mask]
            # 기존 target 값을 0~(num_random_classes-1)로 remap
            mapping = {old: new for new, old in enumerate(sampled)}
            new_targets = [mapping[x.item()] for x in self.targets]
            self.targets = torch.tensor(new_targets, dtype=torch.long)
            # 클래스 이름 업데이트 (예: target 1 -> 'a', target 3 -> 'c' 등)
            self.classes = [valid_letters[t - 1] for t in sorted(sampled)]
        else:
            raise ValueError("split='letters' and num_random_classes must be specified together.")

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except Exception as e:
            print("이미지 변환 중 오류 발생:", e)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.mode = mode

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'

        # self.fpath = os.path.join(root, 'VIL_CORe50')
        self.fpath = os.path.join(root, 'core50_128x128')
        
        if not os.path.isfile(self.fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'core50_128x128')):
            with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error as e:
                        pass

        self.train_session_list = ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']
        self.test_session_list = ['s3', 's7', 's10']
        self.label = [f'o{i}' for i in range(1, 51)]
        
        if not os.path.exists(self.fpath + '/train') and not os.path.exists(self.fpath + '/test'):
            self.split()

        if self.train:
            fpath = self.fpath + '/train'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{s}', transform=transform) for s in self.train_session_list]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)
        else:
            fpath = self.fpath + '/test'
            self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.fpath + '/train'
        test_folder = self.fpath + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        if self.mode not in ['cil', 'joint']:
            for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
                src = os.path.join(self.fpath, s)
                if os.path.exists(os.path.join(train_folder, s)):
                    continue
                move(src, train_folder)
            
            for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(test_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(test_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))
        else:
            for s in tqdm.tqdm(self.train_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(train_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(train_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))

            for s in tqdm.tqdm(self.test_session_list, desc='Preprocessing'):
                for l in self.label:
                    dst = os.path.join(test_folder, l)
                    if not os.path.exists(dst):
                        os.mkdir(os.path.join(test_folder, l))
                    
                    f = glob.glob(os.path.join(self.fpath, s, l, '*.png'))

                    for src in f:
                        move(src, dst)
                rmtree(os.path.join(self.fpath, s))

class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):
        root = os.path.join(root, 'VIL_DomainNet')   
        # root = os.path.join(root, 'DomainNet')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.mode = mode

        self.url = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
        ]

        self.filename = [
            'clipart.zip',
            'infograph.zip',
            'painting.zip',
            'quickdraw.zip',
            'real.zip',
            'sketch.zip'
        ]

        self.train_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt'
        ]

        for u in self.train_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)
        
        self.test_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'
        ]

        for u in self.test_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)

        self.fpath = [os.path.join(self.root, f) for f in self.filename]

        for i in range(len(self.fpath)):
            if not os.path.isfile(self.fpath[i]):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url[i])
                    download_url(self.url[i], root, filename=self.filename[i])

        if not os.path.exists(self.root + '/train') and not os.path.exists(self.root + '/test'):
            for i in range(len(self.fpath)):
                if not os.path.exists(os.path.join(self.root, self.filename[i][:-4])):
                    with zipfile.ZipFile(os.path.join(self.root, self.filename[i]), 'r') as zf:
                        for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename[i]}'):
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
            
            self.split()
        
        if self.train:
            fpath = self.root + '/train'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)
        else:
            fpath = self.root + '/test'
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + '/train'
        test_folder = self.root + '/test'

        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        if self.mode not in ['cil', 'joint']:
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(train_folder, path)

                        move(src, dst)
            
            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(test_folder, path)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))
        else:
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(train_folder, dst)

                        move(src, dst)

            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(test_folder, dst)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))