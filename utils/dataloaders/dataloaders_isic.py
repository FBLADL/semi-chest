from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from skimage import io
from PIL import Image
from torchvision.transforms import transforms
import random
from PIL import ImageFilter
import torch
from utils.gcloud import download_isic_unzip


class ISICDataset(Dataset):
    def __init__(self, root_path, img_filepath, transform):
        super(ISICDataset, self).__init__()
        gr = pd.read_csv(os.path.join(root_path, img_filepath))
        self.root = root_path
        self.imgs = gr['image'].values
        self.gr = gr.iloc[:,1:-1].values.astype(int)
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.imgs[item]
        img_path = os.path.join(self.root, 'train', img_path + '.jpg')
        img = Image.open(img_path).convert('RGB')
        target = self.gr[item]
        img1 = self.transform(img)
        imgs = [img1]
        for i in range(5):
            imgs.append(self.transform(img))
        return imgs, target, item

    def __len__(self):
        return len(self.imgs)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ISICLoader:
    def __init__(self, root_path, batch_size, img_resize=224, gcloud=True):
        self.batch_size = batch_size
        self.root_path = root_path
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if gcloud:
            download_isic_unzip(root_path)
        self.moco_transform = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.CenterCrop(img_resize),
            transforms.RandomAffine(10, translate=(0.02, 0.02)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)  # not strengthened
            ], p=0.8),
            transforms.RandomApply(
                [GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.train_transform = transforms.Compose([

            transforms.RandomResizedCrop((img_resize, img_resize), scale=(0.2, 1.)),
            transforms.RandomAffine(10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def run(self, mode='train', ratio=20, runtime=1):
        if mode == 'train':
            file_path = 'training.csv'
            transform = self.train_transform
        elif mode == 'test':
            file_path = 'testing.csv'
            transform = self.val_transform
        elif mode == 'moco_train':
            file_path = 'training.csv'
            transform = self.moco_transform
        elif mode == 'labeled':
            file_path = f'isic2018_label{ratio}_{runtime}.csv'
            transform = self.train_transform
        elif mode == 'unlabeled':
            file_path = f'isic2018_unlabel{ratio}_{runtime}.csv'
            transform = self.train_transform
        chexpert_dataset = ISICDataset(root_path=self.root_path, img_filepath=file_path,
                                       transform=transform, )
        if mode == 'moco_train':
            sampler = torch.utils.data.distributed.DistributedSampler(
                chexpert_dataset)
        else:
            sampler = None
        loader = DataLoader(dataset=chexpert_dataset, batch_size=self.batch_size,
                            shuffle=True if mode == 'train' else False, pin_memory=True, sampler=sampler,
                            drop_last=True if 'train' in mode else False, num_workers=8)
        return loader, sampler
