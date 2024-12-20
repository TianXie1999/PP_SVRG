import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt

def MNIST_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('data/MNIST', download=True, train=False, transform=transform)
    train_set = datasets.MNIST("data/MNIST", download=True, train=True, transform=transform)
    return train_set, test_set


def MNIST_dataset_sample(p=0.43):
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full datasets
    full_train_set = datasets.MNIST("data/MNIST", download=True, train=True, transform=transform)
    full_test_set = datasets.MNIST("data/MNIST", download=True, train=False, transform=transform)
    
    # Function to sample half of the data with equal class sizes
    def sample_equal_classes(dataset):
        targets = dataset.targets.numpy()  # Get class labels
        classes = torch.unique(dataset.targets).numpy()  # Unique class labels
        num_classes = len(classes)
        samples_per_class = int(len(dataset)*p) // (2 * num_classes)  # p divide the data, split equally across classes
        
        selected_indices = []
        for cls in classes:
            class_indices = (targets == cls).nonzero()[0]  # Indices of this class
            selected_indices.extend(class_indices[:samples_per_class])  # Take required number of samples
        
        return Subset(dataset, selected_indices)
    
    # Create sampled train and test sets
    sampled_train_set = sample_equal_classes(full_train_set)
    sampled_test_set = sample_equal_classes(full_test_set)
    
    return sampled_train_set, sampled_test_set

def CIFAR10_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR10('data/CIFAR10', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR10("data/CIFAR10", download=True, train=True, transform=transform)
    return train_set, test_set

def MNIST_one_layer():
    # Create the nn model
    input_size = 784
    hidden_sizes = [64]
    output_size = 10

    # The non-convex function (model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], output_size),
        nn.LogSoftmax(dim=1))

    return model

def MNIST_two_layers():
    # Create the nn model
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # The non-convex function (model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1))

    return model

class MNIST_ConvNet(nn.Module):
    def __init__(self):
        super(MNIST_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 filters
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)         # Output layer (10 classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)  # Downsample
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)  # Downsample
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10_ConvNet(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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


