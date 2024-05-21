import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PHOCNet import *

from torch.optim.lr_scheduler import StepLR, CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torchvision import datasets, models, transforms

from .build_phoc import phoc
from .dataset import dataset


def get_data(img_dir, transform=None):

    dataset_ = dataset(img_dir, transform)

    return dataset_

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader

def make(config, device="cuda"):

    transforms_train = transforms.Compose([
        transforms.Resize((64, 128), antialias=True),
        transforms.Normalize(mean=[0.445313568], std=[0.26924618])
    ])

    transforms_test = transforms.Compose([
        transforms.Resize((64, 128), antialias=True),
        transforms.Normalize(mean=[0.445313568], std=[0.26924618])
    ])

    train, test = get_data(config.train_dir, transforms_train), get_data(config.test_dir, transforms_test)

    train_loader = make_loader(train, config.batch_size)
    test_loader = make_loader(test, config.batch_size)

    # Make the model
    model = PHOCNet(n_out = train[0][1].shape[0], input_channels = 1).to(device)
    model.apply(init_weights_model)
    """model = models.resnet18(pretrained=True) 
    set_parameter_requires_grad(model,True)
    model.fc = nn.Sequential(nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, train[0][1].shape[0]))
    model=model.to(device)"""
    
    pos_weight = torch.tensor(create_weights(config.train_dir)).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction = 'mean', pos_weight=pos_weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.3)
    #scheduler = CosineAnnealingLR(optimizer, T_max=10)
    #scheduler = ReduceLROnPlateau(optimizer, patience = 2, factor = 0.1)
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler

def init_weights_model(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def create_weights(file_words):
    paths = os.listdir(file_words)
    list_of_words = [path.split("_")[-1].split(".")[0] for path in paths]
    list_of_words = [l[:-1] for l in list_of_words]
    phoc_representations = phoc(list_of_words)
    suma = np.sum(phoc_representations, axis=0)
    weights = phoc_representations.shape[0]/(suma+1e-6) 
    weights = (1 + (weights/max(weights)))*5
    return weights

def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False