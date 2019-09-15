from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import numpy as np
import glob
import random
import torch

class npy_train(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms=transforms
        self.x_train = np.load(path+'XTRAIN_channels_8_torch.npy')
        self.y_train = np.load(path+'YTRAIN_channels_8_torch.npy')
        # self.x_train=(self.x_train-self.transforms[0])/self.transforms[1]
        # self.y_train=(self.y_train-self.transforms[2])/self.transforms[3]

    def __getitem__(self, index):
        x_sample, y_sample = self.x_train[index], self.y_train[index]

        return x_sample, y_sample

    def __len__(self):
        return len(self.x_train)

class npy_valid(Dataset):
    def __init__(self, path, transforms=None):
        # self.classes = os.listdir(path)
        self.path = path
        self.transforms=transforms
        self.x_train = np.load(path+'XVALID_channels_8_torch.npy')
        self.y_train = np.load(path+ 'YVALID_channels_8_torch.npy')

    def __getitem__(self, index):
        x_sample, y_sample = self.x_train[index], self.y_train[index]

        return x_sample, y_sample

    def __len__(self):
        return len(self.x_train)

class npy_test(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms=transforms
        self.x_train = np.load(path+'XTEST_channels_8_torch.npy')
        self.y_train = np.load(path+ 'YTEST_channels_8_torch.npy')

        
    def __getitem__(self, index):
        x_sample, y_sample = self.x_train[index], self.y_train[index]

        return x_sample, y_sample

    def __len__(self):
        return len(self.x_train)