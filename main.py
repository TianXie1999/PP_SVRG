import torch
from torch import nn 
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
from optim import SGD_Simple, SVRG_k, SVRG_Snapshot
from utils.dataset import MNIST_dataset, MNIST_dataset_sample, CIFAR10_dataset
from models import MNIST_two_layers, MNIST_one_layer, MNIST_ConvNet, CIFAR10_ConvNet
from utils.utils import AverageCalculator, accuracy, groupwise_weights
from utils.parser import get_args



OUTPUT_DIR = "outputs"


from utils.engine import train_one_epoch

def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(2)
        device = 'cuda'
    print(f"Using device: {device}")
    return device

def load_dataset(args):
    if args.dataset == "MNIST":
        if args.ratio < 1:
            train_set, val_set = MNIST_dataset_sample(p=args.ratio)
        else:
            train_set, val_set = MNIST_dataset()
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset()
    else:
        raise ValueError("Unknown dataset")
    return train_set, val_set

def initialize_model(args, device):
    model_dict = {
        "MNIST_one_layer": MNIST_one_layer,
        "MNIST_two_layers": MNIST_two_layers,
        "MNIST_ConvNet": MNIST_ConvNet,
        "CIFAR10_convnet": CIFAR10_ConvNet
    }
    NN_model = model_dict.get(args.nn_model)
    if NN_model is None:
        raise ValueError("Unknown model")
    model = NN_model().to(device)
    model_snapshot = NN_model().to(device) if args.optimizer == 'SVRG' else None
    return model, model_snapshot

def initialize_optimizer(args, model, model_snapshot):
    optimizers = {
        "SGD": SGD_Simple,
        "SVRG": SVRG_k
    }
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters()) if args.optimizer == 'SVRG' else None
    return optimizer, optimizer_snapshot

def get_loss_fn(loss_type):
    loss_fns = {
        'NLLLoss': nn.NLLLoss,
        'CrossEntropyLoss': nn.CrossEntropyLoss
    }
    return loss_fns.get(loss_type, nn.CrossEntropyLoss)

def setup_output_directory(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{timestamp}_{args.optimizer}_{args.nn_model}_Temperature{str(args.temperature)}_lr{str(args.lr)}"
    if args.exp_name != "":
        model_name = f"{args.exp_name}_{model_name}"
    log_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    return log_dir

def train_model(args, model, model_snapshot, optimizer, optimizer_snapshot, train_loader, 
                train_loader_large, loss_fn, device):
    
    
    temperature = args.temperature
    start_weights = groupwise_weights(model, train_loader_large, loss_fn(reduction='none'),
                                      beta=temperature, device=device)
    loss = AverageCalculator()
    acc = AverageCalculator()
    grad = AverageCalculator()

    train_loss_all, train_acc_all, weights_all, grads_all = [], [], [], []

    for epoch in range(args.n_epoch):
        t0 = time.time()

        
        train_loss, train_acc, grads, new_weights = train_one_epoch(
                model, optimizer, train_loader, train_loader_large, start_weights, loss, 
                acc, grad, loss_fn, model_snapshot, optimizer_snapshot, temperature, 
                optimize=args.optimizer, device=device
            )


        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        weights_all.append(new_weights)
        grads_all.append(grads)

        if epoch % args.print_every == 0:
            print(f"Epoch {epoch} / {args.n_epoch}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")

        start_weights = new_weights

        if (epoch + 1) % 1 == 0:
            np.savez(os.path.join(log_dir, 'train_stats.npz'),
                     train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all), weights=np.array(weights_all), grads=np.array(grads_all))

    open(os.path.join(log_dir, 'done'), 'a').close()

if __name__ == "__main__":
    device = get_device()
    args = get_args()
    

    if args.optimizer not in ['SGD', 'SVRG']:
        raise ValueError("--optimizer must be 'SGD' or 'SVRG'.")
    print(vars(args))

    # load the data
    train_set, val_set = load_dataset(args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_loader_large = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    loss_fn = get_loss_fn(args.loss_type)
    

    # initialize the model
    model, model_snapshot = initialize_model(args, device)
    # initialize the optimizer
    optimizer, optimizer_snapshot = initialize_optimizer(args, model, model_snapshot)

    # setup output directory
    log_dir = setup_output_directory(args)

    train_model(args, model, model_snapshot, optimizer, optimizer_snapshot, train_loader, 
                train_loader_large, loss_fn, device)
            