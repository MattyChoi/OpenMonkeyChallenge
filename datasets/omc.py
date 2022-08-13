import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import json
import numpy as np
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset
from utils.preprocess import crop, gen_hmaps


class OMC(Dataset):
    """
    Dataset for Open Monkey Challenge Dataset
    """
    def __init__(self, data_dir="", transform=None):
        super(OMC, self).__init__()

        self.transform = transform
        self.data_dir = data_dir

        # get the training data
        dir = os.path.join(data_dir, 'train_annotation.json')

        with open(dir) as f:
            dic = json.load(f)
            self.feature_list = [item for item in dic['data']]


    def __getitem__(self, index):
        features = self.feature_list[index]
        input_shape = (256,256)
        hmap_shape = (64,64)

        img_folder_dir = os.path.join(self.data_dir, 'train')
        img_dir = os.path.join(img_folder_dir, features['file'])
        img = mpimg.imread(img_dir)

        # generate crop image
        #print(img)
        img_crop, pts_crop = crop(img, features)
        pts_crop = np.array(pts_crop)
        
        train_img = np.transpose(img_crop, (2,0,1))/255.0
        train_heatmaps = gen_hmaps(np.zeros(hmap_shape), pts_crop/4)
        
        train_heatmaps = np.transpose(train_heatmaps, (2,0,1))

        return torch.Tensor(train_img), torch.Tensor(train_heatmaps)


    def __len__(self):
        return len(self.feature_list)


    def collate_fn(self, batch):
        imgs, heatmaps = list(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)

        return imgs, heatmaps


class OMC_Test(Dataset):
    """
    Test Dataset for Open Monkey Challenge Dataset
    """
    def __init__(self, data_dir="", transform=None):
        super(OMC_Test, self).__init__()

        self.transform = transform
        self.data_dir = data_dir

        # take the validation data
        dir = os.path.join(data_dir, 'val_annotation.json')

        with open(dir) as f:
            dic = json.load(f)
            self.feature_list = [item for item in dic['data']]


    def __getitem__(self, index):
        features = self.feature_list[index]
        input_shape = (256,256)
        hmap_shape = (64,64)

        img_folder_dir = os.path.join(self.data_dir, 'val')
        img_dir = os.path.join(img_folder_dir, features['file'])
        img = mpimg.imread(img_dir)

        # generate crop image
        #print(img)
        img_crop, pts_crop = crop(img, features)
        pts_crop = np.array(pts_crop)
        
        train_img = np.transpose(img_crop, (2,0,1))/255.0
        train_heatmaps = gen_hmaps(np.zeros(hmap_shape), pts_crop/4)
        
        train_heatmaps = np.transpose(train_heatmaps, (2,0,1))

        return torch.Tensor(train_img), torch.Tensor(train_heatmaps)


    def __len__(self):
        return len(self.feature_list)


    def collate_fn(self, batch):
        imgs, heatmaps = list(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)

        return imgs, heatmaps