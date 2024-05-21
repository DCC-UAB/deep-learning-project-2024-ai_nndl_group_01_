from tqdm.auto import tqdm
import wandb
from test import test
"""import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"""
from utils.wandb_logs import *
from utils.predict_with_PHOC import load_model
import torch
import os

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, config, device = "cuda"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    example_ct = 0  # number of examples seen
    batch_ct = 0
    model_phoc = load_model()
    for epoch in tqdm(range(config.epochs)):
        model.train()
        train_loss = 0
        for _, (images, phoc_labels, _) in enumerate(train_loader):

            loss = train_batch(images, phoc_labels, model, optimizer, criterion, device)
            train_loss += loss.item()
            example_ct +=  len(images)
            batch_ct += 1

            #Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss.item(), example_ct)
        
        loss_test = test(model, test_loader, train_loader, epoch, criterion, model_phoc, device)
        
        scheduler.step(loss_test)
        print(scheduler._last_lr)
        torch.save(model.state_dict(), os.path.join(config.save_model, f"PHOCNET{epoch}.pt"))
    return model

def train_batch(images, labels, model, optimizer, criterion, device="cuda"):
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs.float(), labels.float())
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss