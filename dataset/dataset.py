import os 
import numpy as np
import pandas as pd
import torch
import torch.utils
from torch.utils.data import Subset, Dataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms
def MNIST_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    test_set = datasets.MNIST('data/MNIST', download=True, train=False, transform=transform)
    train_set = datasets.MNIST("data/MNIST", download=True, train=True, transform=transform)
    return train_set, test_set


def MNIST_dataset_sample(p=0.43):
    """sample p fraction of the data with equal class sizes

    Args:
        p (float, optional): fraction of the data to sample. Defaults to 0.43.

    Returns:
        List[Subset]: sampled train and test sets
    """
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
def balanced_subset(dataset, p_sample):
    targets = np.array(dataset.targets)
    unique_classes = np.unique(targets)
    indices = []

    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        sampled_count = int(len(cls_indices) * p_sample)
        sampled_indices = np.random.choice(cls_indices, sampled_count, replace=False)
        indices.extend(sampled_indices)

    return Subset(dataset, indices)
        
        
def longtail_subset(dataset, alpha=0.8):
    """
    make a balanced dataset into a longtail dataset 
    Args:
        dataset (Dataset): dataset to be sampled
        alpha (float): longtail parameter, we keep 1 * alpha ^ i of the data in class i 
    """ 
    targets = np.array(dataset.targets)
    unique_classes = np.unique(targets)
    indices = []

    for i, cls in enumerate(unique_classes):
        cls_indices = np.where(targets == cls)[0]
        sampled_count = int(len(cls_indices) * (1 * alpha ** i))
        sampled_indices = np.random.choice(cls_indices, sampled_count, replace=False)
        indices.extend(sampled_indices)
    # showcase each class size
    class_sizes = {cls: len(np.where(targets[indices] == cls)[0]) for cls in unique_classes}
    print("class sizes: ", class_sizes)
    return Subset(dataset, indices)

def CIFAR10_dataset(p_sample=1):
    """Get CIFAR10 dataset, p_sample is the fraction of the data to sample.

    Args:
        p_sample (float, optional): Fraction of the data to sample (0 < p_sample <= 1). Defaults to 1.

    Returns:
        tuple: (train_set, test_set), where each is a torch.utils.data.Dataset.
    """
    if not os.path.isdir("data"):
        os.mkdir("data")

    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load datasets
    test_set = datasets.CIFAR10('data/CIFAR10', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR10("data/CIFAR10", download=True, train=True, transform=transform)

    # If p_sample < 1, subsample the dataset
    if p_sample < 1:
        train_set = balanced_subset(train_set, p_sample)

    return train_set, test_set

def CIFAR100_dataset(p_sample=1):
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR100('data/CIFAR100', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR100("data/CIFAR100", download=True, train=True, transform=transform)
    if p_sample < 1:
        train_set = balanced_subset(train_set, p_sample)
    
    return train_set, test_set

def sample_by_class(dataset, type='balanced', p_sample=1):
    targets = np.array(dataset.targets)
    unique_classes = np.unique(targets)
    indices = []

    if type == 'balanced':
        for cls in unique_classes:
            cls_indices = np.where(targets == cls)[0]
            sampled_count = int(len(cls_indices) * p_sample)
            sampled_indices = np.random.choice(cls_indices, sampled_count, replace=False)
            indices.extend(sampled_indices)

    elif type == 'longtail':
        pass 

def processing_credit_dataset(data_path):
    # delete the data with NA
    data_frame = pd.read_csv(data_path)
    data_frame = data_frame.dropna()
    data = np.array(data_frame)
    data, labels = data[:, 2:], data[:, 1]
    data[:,[1,2,-5,-3]] = data[:,[-3,-5,1,2]]
    return data, labels

    
class CreditDataset(Dataset):
    def __init__(self, num_samples=10000, data_type='train'):
        super().__init__()
        
        path = 'data/givemesomecredit/'
        if data_type == 'train':
            data_path = f'{path}cs-training.csv'
            
        else:
            data_path = f'{path}cs-test.csv'
        
        data, labels = processing_credit_dataset(data_path)
        
        if num_samples > len(labels):
            raise ValueError("Requested number of samples exceeds available samples in the dataset")
        
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        # the labels are two class, split 10000 into two 5000 samples
        
        class0_idx = np.where(labels == 0)[0]
        class1_idx = np.where(labels == 1)[0]

        half = num_samples // 2
        print("class0_idx: ", len(class0_idx), "class1_idx: ", len(class1_idx))
        print("half: ", half)

        if len(class0_idx) < half or len(class1_idx) < half:
            raise ValueError("Not enough samples in one of the classes to balance the dataset")

        
        sampled_class0 = np.random.choice(class0_idx, half, replace=False)
        sampled_class1 = np.random.choice(class1_idx, half, replace=False)

        
        combined_idx = np.concatenate([sampled_class0, sampled_class1])
        np.random.shuffle(combined_idx)

        
        self.data = torch.tensor(data[combined_idx], dtype=torch.float32)
        self._standardize_data()
        self.labels = torch.tensor(labels[combined_idx], dtype=torch.int64) 
            
    def _standardize_data(self):
        mean = self.data.mean(dim=0)
        std = self.data.std(dim=0)
        self.data = (self.data - mean) / std
        self.data = torch.nan_to_num(self.data, nan=0.0, posinf=1.0, neginf=-1.0)
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
def get_credit_dataset(p_sample=1):
    dataset = CreditDataset(num_samples=int(10000*p_sample), data_type='train')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)//2, len(dataset) - len(dataset)//2])
    return train_dataset, val_dataset

def load_dataset(args):
    if args.dataset == "MNIST":
        if args.ratio < 1:
            train_set, val_set = MNIST_dataset_sample(p=args.ratio)
        else:
            train_set, val_set = MNIST_dataset()
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset(p_sample=args.ratio)
    elif args.dataset == "CIFAR100":
        train_set, val_set = CIFAR100_dataset(p_sample=args.ratio)
    elif args.dataset == "credit":
        train_set, val_set = get_credit_dataset(p_sample=args.ratio)
    else:
        raise ValueError("Unknown dataset")
    return train_set, val_set