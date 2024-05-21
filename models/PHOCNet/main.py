import wandb
import torch
from train import train
from utils.utils import *
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from params import *

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg:dict) -> None:
    with wandb.init(project="PHOCnet_BN", config=cfg):
        config = wandb.config

        model, train_loader, test_loader, criterion, optimizer, scheduler = make(config, device)
        
        model = train(model, train_loader, test_loader, criterion, optimizer, scheduler, config, device)

        return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        train_dir=train_images+"/",
        test_dir=test_images+"/",
        epochs=8,
        batch_size= 8,
        learning_rate=0.01,
        save_model = saved_model_phocnet+"/")
    model = model_pipeline(config)      