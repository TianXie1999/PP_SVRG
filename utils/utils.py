import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt



def accuracy(yhat, labels):
    _, indices = yhat.max(1)
    return (indices == labels).sum().data.item() / float(len(labels))

class AverageCalculator():
    def __init__(self):
        self.reset() 
    
    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, val, n=1):
        assert(n > 0)
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / float(self.count)

def plot_train_stats(train_loss_1, train_acc_1, train_grad_norms_1, train_loss_2, train_acc_2, train_grad_norms_2, directory, acc_low=0):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,2.5), sharey='row')
    axes[0].plot(np.array(train_loss_1), label="SGD")
    axes[0].plot(np.array(train_loss_2), label="SVRG")
    axes[0].set_title("Train loss")
    axes[0].legend()
    axes[1].plot(np.array(train_acc_1), label="SGD")
    axes[1].plot(np.array(train_acc_2), label="SVRG")
    axes[1].set_ylim(acc_low, 1)
    axes[1].set_title("Train Accuracy")
    axes[1].legend()
    axes[2].plot(np.array(train_grad_norms_1), label="SGD")
    axes[2].plot(np.array(train_grad_norms_2), label="SVRG")
    axes[2].set_title("Train Gradient Norms")
    # set a log yticks
    axes[2].set_yscale('log')
    axes[2].legend()
    # add a global x axis
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'train_stats.pdf'))
    plt.close()

def groupwise_weights(model_k, train_loader_large, loss_fn, beta=0.5, device='cpu'):
    """
    first calculate the loss of each label group wrt model_k
    then calculate each group's weight as softmax of -beta * loss
    also calculate the average gradient norm
    return a dictionary with keys = label, values = weights
    """
    model_k.eval()
    group_loss = {}
    for images, labels in train_loader_large:
        images = images.to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        # loss for each sample without averaging
        loss_iter = loss_fn(yhat, labels)
        loss_iter = loss_iter.view(-1)  # Ensure loss_iter is a 1D tensor
        unique_labels = torch.unique(labels)
        # if torch.abs(loss_iter).sum() > 50000:
        #     print(len(loss_iter))
        #     print("Loss iter: ", loss_iter)
        #     print("Labels: ", labels)
        #     print("Unique labels: ", unique_labels)
        #     print("yhat:", yhat)
        for i in range(len(unique_labels)):
            label = unique_labels[i].item()
            group_loss[label] = loss_iter[labels == label].mean()  
    # print("Group losses: ", group_loss)
    max_loss = max(group_loss.values())
    
    group_weights = {
        label: np.exp(-beta * (group_loss[label] - max_loss).item()) for label in group_loss
        }
    total = sum(group_weights.values())
    #clip the weights to [0.25, 4]
    # for label in group_weights:
    #     group_weights[label] = min(2, max(0.5, group_weights[label]))
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
    return loss_iter, yhat

def update_weights(model, train_loader_large, loss_fn, beta, device):
    return groupwise_weights(model, train_loader_large, loss_fn, beta=beta, device=device)

def log_metrics(loss_iter, yhat, labels, loss, acc):
    acc_iter = accuracy(yhat, labels)
    loss.update(loss_iter.data.item())
    acc.update(acc_iter)
    print(f"loss: {loss_iter.data.item()}, acc: {acc_iter}")

def calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()
    for images, labels in train_loader:
        loss_iter, _ = calculate_loss(model, images, labels, start_weights, loss_fn, device)
        loss_iter.backward()
    
    full_grd = torch.cat([param.grad.view(-1) for param in model.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    return g