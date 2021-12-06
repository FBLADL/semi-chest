import os
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import ImageFilter
from skimage import io
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


Labels = {'No Finding': 14, 'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltration': 3, 'Mass': 4,
          'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7,
          'Consolidation': 8, 'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}
mlb = MultiLabelBinarizer(
    classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ChestDataset(Dataset):
    def __init__(self, root_dir, transform, mode, runtime=1, ratio=100,k_crop=5) -> None:
        self.transform = transform
        self.root_dir = root_dir
        self.mode = mode
        self.k_crops = k_crop

        gr_path = os.path.join(root_dir, 'Data_Entry_2017.csv')
        gr = pd.read_csv(gr_path, index_col=0)
        gr = gr.to_dict()['Finding Labels']

        read_img_list = 'test_list.txt' if self.mode == 'test' else 'valid_list.txt' if self.mode == 'val' else 'train_val_list_{}_{}.txt'.format(
            ratio, runtime) if self.mode == 'sup_train' else 'train_val_list.txt'
        img_path_list = os.path.join(root_dir, read_img_list)
        # img_all_path_list = os.path.join(root_dir,'train_val_list.txt')
        with open(img_path_list) as f:
            names = f.read().splitlines()
        # with open(img_all_path_list) as f:
        #     all_names = f.read().splitlines()
        self.imgs = np.asarray([x for x in names])
        # self.all_imgs = np.asarray([x for x in all_names])
        # self.unlabel_imgs = self.all_imgs - self.imgs
        gr = np.asarray([gr[i] for i in self.imgs])

        self.gr = np.zeros((gr.shape[0], 15))
        for idx, i in enumerate(gr):
            target = i.split('|')
            binary_result = mlb.fit_transform(
                [[Labels[i] for i in target]]).squeeze()
            self.gr[idx] = binary_result

    def __getitem__(self, item):
        img = io.imread(os.path.join(self.root_dir, 'data', self.imgs[item]))
        img = Image.fromarray(img).convert('RGB')
        img1 = self.transform(img)
        # img1 = self.transform(image=img)['image']

        target = self.gr[item]
        imgs = [img1]
        if self.mode == 'moco_train':
            for i in range(self.k_crops):
                imgs.append(self.transform(img))
            # img2 = self.transform(img)
            # img2 = self.transform(image=img)['image']
            return imgs, target
        else:
            return img1, target

    def __len__(self):
        return self.imgs.shape[0]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ChestDataloader():
    def __init__(self, batch_size=128, num_workers=8, img_resize=512, root_dir=None, gc_cloud=False,k_crop=5):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_resize = img_resize
        self.k_crop = k_crop
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.root_dir = root_dir

        # strong_aug = randaug.CLSAAug()
        self.transform_moco_train = transforms.Compose([
            transforms.RandomResizedCrop(img_resize, scale=(0.2, 1.)),

            transforms.RandomApply([
                transforms.RandomAffine(10)
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)  # not strengthened
            ], p=0.8),
            transforms.RandomApply(
                [GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, inplace=True),
        ])

        self.transform_sup_train = transforms.Compose([
            transforms.RandomResizedCrop(img_resize, scale=(0.2, 1.)),
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def run(self, mode, exp=None, ratio=100, runtime=1):
        if exp:
            use_transform = self.transform_moco_train_exp if mode == 'moco_train' else self.transform_sup_train if mode == 'sup_train' else self.transform_test
        else:
            use_transform = self.transform_moco_train if mode == 'moco_train' else self.transform_sup_train if mode == 'sup_train' else self.transform_test

        all_dataset = ChestDataset(
            root_dir=self.root_dir, transform=use_transform, mode=mode, ratio=ratio, runtime=runtime,k_crop=self.k_crop)
        if mode == 'moco_train':
            sampler = torch.utils.data.distributed.DistributedSampler(
                all_dataset)
        else:
            sampler = None
        batch_size = self.batch_size
        loader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False if mode == 'moco_train' else True,
                            sampler=sampler, num_workers=self.num_workers,
                            pin_memory=True,
                            drop_last=True if 'train' in mode else False)

        return loader, sampler
