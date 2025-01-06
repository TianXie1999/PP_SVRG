import os 
import torch
from torch.utils.data import Subset
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

def CIFAR10_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR10('data/CIFAR10', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR10("data/CIFAR10", download=True, train=True, transform=transform)
    return train_set, test_set

def CIFAR100_dataset():
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Download MNIST dataset and set the valset as the test test
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = datasets.CIFAR100('data/CIFAR100', download=True, train=False, transform=transform)
    train_set = datasets.CIFAR100("data/CIFAR100", download=True, train=True, transform=transform)
    return train_set, test_set


def load_dataset(args):
    if args.dataset == "MNIST":
        if args.ratio < 1:
            train_set, val_set = MNIST_dataset_sample(p=args.ratio)
        else:
            train_set, val_set = MNIST_dataset()
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset()
    elif args.dataset == "CIFAR100":
        train_set, val_set = CIFAR100_dataset()
    else:
        raise ValueError("Unknown dataset")
    return train_set, val_set