import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from train import train
from test import test
from models import PHOCNet
from utils import make_dataloaders  # Assume you have a utility function for dataloaders

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg: dict) -> None:
    # tell wandb to get started
    with wandb.init(project="phocnet-text-recognition", config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model = PHOCNet(n_out=config.classes).to(device)
        train_loader, test_loader = make_dataloaders(config)
        criterion = nn.BCELoss()  # or another appropriate loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Train the model
        train(model, train_loader, criterion, optimizer, config)

        # Test its performance
        test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=5,
        batch_size=128,
        learning_rate=1e-3,
        dataset="MJSynth",  # Your dataset
        architecture="PHOCNet"
    )

    model = model_pipeline(config)
