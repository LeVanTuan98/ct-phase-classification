# -*- coding: utf-8 -*-
"""
Created on 3/09/2020 8:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import random


# Third party imports
import nibabel as nib
import SimpleITK as sitk
import cv2
import torch
import torchvision
import monai.transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

 
class NiiDataset(Dataset):

    def __init__(self, csv_files='', input_shape=(224, 224), 
                 hu_range=[-200, 200], n_slices=10, scale_intensity=[0, 1], 
                 label_list=['Non_Contrast', 'Arterial_Phase', 'Venous_Phase', 'Delayed_Phase'], 
                 root_dir='', csv_columns=['ct_liver_cropped', 'phase_name'], select_random=False, **kwargs):
        self.input_shape = input_shape
        self.root_dir = root_dir
        self.n_slices = n_slices
        self.hu_range = hu_range
        self.select_random = select_random
#         print(hu_range, select_random)
        # print(input_shape, root_dir, n_slices, hu_range)
        # Read data from csv files
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        # Read the first csv file
        df = pd.read_csv(csv_files[0])
        df = df[csv_columns]
        #print(len(df))
        
        # Read the rest csv files
        for csv_file in csv_files[1:]:
            df_added = pd.read_csv(csv_file)
            df_added = df_added[csv_columns]
            df = df.append(df_added)
        df['phase_id'] = df[csv_columns[1]].apply(lambda x: label_list.index(x))
        if self.root_dir != None and self.root_dir != '':
            df[csv_columns[0]] = df[csv_columns[0]].apply(lambda x: f"{root_dir}/{x}")
        df = df[[csv_columns[0], 'phase_id']]
            
        # Convert df into a list
        self.imgs = df.values.tolist()
    
        # Set the augmentation
        self.transforms = monai.transforms.Compose([monai.transforms.Resize(input_shape, size_mode='all'),
                                                    monai.transforms.ToTensor(dtype=torch.float32)])
#         self.transforms = monai.transforms.Compose([monai.transforms.Resize(input_shape, size_mode='all'),
#                                                     monai.transforms.ScaleIntensity(minv=scale_intensity[0],
#                                                                                     maxv=scale_intensity[1]),
#                                                     monai.transforms.ToTensor(dtype=torch.float32)])

    def __getitem__(self, index):
        sample = self.imgs[index]
        # print(sample)
        # Read the data using simple itk to obtain the size of (n_slices x H x W)
        data = sitk.GetArrayFromImage(sitk.ReadImage(sample[0])) 
        # print(data.shape, data.shape[0], self.n_slices)
        
        # Select randomly n_slices from the volume
        # data = data[random.sample(range(data.shape[0]), self.n_slices), ...]
        if self.select_random:
            data = data[np.random.choice(data.shape[0], self.n_slices, replace=data.shape[0]<self.n_slices), ...]
        else:
            data = data[np.linspace(0, data.shape[0] - 1, self.n_slices, dtype=int), ...]
    
        data = np.clip(data, self.hu_range[0], self.hu_range[1]) # clip the hu_range
        
        # Transformn the data
        data = self.transforms(data)
        label = np.float32(sample[1])

        return data, label

    def __len__(self):
        return len(self.imgs)


    
class ImageDataset(Dataset):

    def __init__(self, csv_files, phase='train', 
                 input_shape=(128, 128), root_dir='',
                 csv_columns=['path', 'Covid', 'source', 'source_group']):
        self.phase = phase
        self.input_shape = input_shape
        self.root_dir = root_dir
        
        # Read data from csv files
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        # Read the first csv file
        df = pd.read_csv(csv_files[0])
        #print(len(df), df.columns)
        if 'seg_ok' in df.columns: # remove the samples not good lung segmentation
            df = df.drop(np.where(df['seg_ok'] == 0)[0])
        df = df[csv_columns]
        #print(len(df))
        
        # Read the rest csv files
        for csv_file in csv_files[1:]:
            df_added = pd.read_csv(csv_file)
            df_added = df_added[csv_columns]
            df = df.append(df_added)
            
        # Convert df into a list
        self.imgs = df.values.tolist()

        # Set the augmentation
        if self.phase == 'train':
            self.transforms = T.Compose([
                    T.ToPILImage(),
                    T.Resize(self.input_shape),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(brightness=(0.2), contrast=(0.85, 1.0))], p=0.4),
                    T.ToTensor(),
                    Cutout(5, self.input_shape[-1]//4)
                ])
        else:
            self.transforms = T.Compose([
                T.ToPILImage(),
                T.Resize(self.input_shape),
                T.ToTensor()
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]

        img_path = sample[0]
        data = cv2.imread(img_path)
        if self.root_dir != None and self.root_dir != '':
            data = data * cv2.imread(self.root_dir + img_path)
            
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = self.transforms(data)
        label = np.float32(sample[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, p=0.15):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if not np.random.choice([True, False], p=[self.p, 1 - self.p]):
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        n_holes = np.random.randint(self.n_holes + 1)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            length = np.random.randint(5, self.length + 1)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = ImageDataset(csv_files='/vinbrain/huyta/COVID_wave_4/csv/source_ablation/Ho Chi Minh/train_no_past.csv',
                           phase='train',
                           input_shape=(320, 320))

    trainloader = DataLoader(dataset, batch_size=16)
    for i, (data, label) in enumerate(trainloader):
        img = torchvision.utils.make_grid(data).numpy()
        print(data.shape, img.shape)
        img = np.transpose(img, (1, 2, 0))
        
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]
        plt.imshow(img)
        plt.show()
        if i > 10:
            break
