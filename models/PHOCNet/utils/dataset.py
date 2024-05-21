from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import os

from .build_phoc import phoc

class dataset(Dataset):
    def __init__(self, img_dir, transform = None):
        
        self.paths = os.listdir(img_dir)
        #self.paths = self.paths[:int(len(self.paths)*0.01)]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):

        return len(self.paths)
    
    def __getitem__(self, idx):

        path = self.img_dir + self.paths[idx]

        img = read_image(path)
        img = img.to(torch.float32)
        if self.transform != None:
            img = self.transform(img)
        img /= 255

        #img = img.expand(3,-1,-1)
        
        word = self.paths[idx].split("_")[-1].split(".")[0]
        target = phoc(word)
        target = target.reshape(target.shape[1])

        return img, target, word