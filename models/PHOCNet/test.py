import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.wandb_logs import *
from utils.predict_with_PHOC import predict_with_PHOC

import editdistance

def test(model, test_loader, train_loader, epoch, criterion, model_phoc, device="cuda", save:bool= True):
    # Run the model on some test examples
    model.eval()
    with torch.no_grad():
        loss_test = 0
        loss_train = 0
        correct_test = 0
        correct_train = 0
        edit_test = 0
        edit_train = 0
        test_count = 0
        train_count = 0
        
        for i, (images, phoc_labels, text_labels) in enumerate(test_loader):
            images, phoc_labels = images.to(device), phoc_labels.to(device)
            test_count += len(images)
            outputs = model(images)
            loss_test += criterion(outputs, phoc_labels.float()) 
            predicted_labels = predict_with_PHOC(torch.sigmoid(outputs).cpu().numpy(), model_phoc)
            correct_test += (predicted_labels == text_labels).sum().item()
            edit_test += sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])
            if i == 0:
                log_images(images, predicted_labels, text_labels[:5], epoch, "Test")
                
        for i, (images, phoc_labels, text_labels) in enumerate(train_loader):
            images, phoc_labels = images.to(device), phoc_labels.to(device)
            train_count += len(images)
            outputs = model(images)
            loss_train += criterion(outputs, phoc_labels.float()) 
            predicted_labels = predict_with_PHOC(torch.sigmoid(outputs).cpu().numpy(), model_phoc)
            correct_train += (predicted_labels == text_labels).sum().item()
            edit_train += sum([editdistance.eval(p,t) for p,t in zip(predicted_labels, text_labels)])
            if i == 0:
                log_images(images, predicted_labels, text_labels[:5], epoch, "Train")
            if i == 150:
               break

        loss_test = loss_test/len(test_loader)
        loss_train = loss_train/(i+1)    
        accuracy_test = correct_test/test_count
        accuracy_train = correct_train/train_count
        edit_test = edit_test/test_count
        edit_train = edit_train/train_count 

        train_test_log(loss_test, loss_train, accuracy_test, accuracy_train, edit_test, edit_train, epoch)
    return loss_test