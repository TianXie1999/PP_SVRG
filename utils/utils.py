import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt

def groupwise_weights(
    model_k, train_loader_large, loss_fn, beta=1, device='cpu'
    # model_k, train_loader_large, loss_fn, beta=0.5, device='cpu'
    ):
    """
    first calculate the loss of each label group wrt model_k
    then calculate each group's weight as softmax of -beta * loss
    also calculate the average gradient norm
    return a dictionary with keys = label, values = weights
    """
    model_k.eval()
    group_loss = {}
    group_weights = {}

    for images, labels in train_loader_large:
        images = images.to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        # loss for each sample without averaging
        loss_iter = loss_fn(reduction='none')(yhat, labels)
        loss_iter = loss_iter.view(-1)  # Ensure loss_iter is a 1D tensor
        unique_labels = torch.unique(labels)
        for i in range(len(unique_labels)):
            label = unique_labels[i].item()
            if label not in group_loss:
                group_loss[label] = 0.0
                group_weights[label] = 0
            
            group_loss[label] += loss_iter[labels == label].mean()
            group_weights[label] += (labels == label).sum().item()
            
    for label in group_loss:
        group_loss[label] /= group_weights[label]        
    
    # print("Group losses: ", group_loss)
    max_loss = max(group_loss.values())
    # calculate softmax of -beta * loss
    group_weights = {
        label: np.exp(-beta * (group_loss[label] - max_loss).item()) for label in group_loss
        }
    total = sum(group_weights.values())
    
    
    
    for label in group_weights:
        group_weights[label] /= total
        group_weights[label] *= len(group_weights)  # scale to the number of groups
    # print("Group weights: ", group_weights)
    
    return group_weights

def calculate_loss(model, images, labels, weights, loss_fn, device):
    images = images.to(device)
    labels = labels.to(device)
    label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
    yhat = model(images)
    loss_iter = loss_fn(weight=label_weights)(yhat, labels)
    # loss_iter = loss_fn()(yhat, labels)
    return loss_iter, yhat

def update_weights(model, train_loader_large, loss_fn, beta, device):
    return groupwise_weights(model, train_loader_large, loss_fn, beta=beta, device=device)

def log_metrics(loss_iter, acc, metric, iter):
    metric['loss'].update(loss_iter.data.item())
    metric['acc'].update(acc)
    print(f"iteration{iter}:loss: {loss_iter.data.item()}, acc: {acc}")

def calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()
    for images, labels in train_loader:
        loss_iter, _ = calculate_loss(model, images, labels, start_weights, loss_fn, device)
        loss_iter.backward()
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad /= len(train_loader)
    
    full_grd = torch.cat([param.grad.view(-1) for param in model.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    return g