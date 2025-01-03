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



def train_epoch_SGD(model, optimizer, train_loader, train_loader_large, start_weights, loss, acc, grad, loss_fn, temperature=0.5):
    model.train()
    # calculate the mean gradient
    optimizer.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        yhat = model(images)
        labels = labels.to(device)
        label_weights = torch.tensor([start_weights[label] for label in start_weights.keys()], dtype=torch.float32).to(device)
        loss_iter = loss_fn(weight=label_weights)(yhat, labels) / len(train_loader)
        loss_iter.backward()
    # print(*model.parameters())
    full_grd = torch.cat([param.grad.view(-1) for param in model.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    grad.update(g)

    weights = start_weights

    for i, (images, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print("Iteration: ", i)
        images = images.to(device)
        yhat = model(images)
        labels = labels.to(device)
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
    
        loss_iter = loss_fn(weight=label_weights)(yhat, labels)
        # loss_iter = loss_fn(yhat, labels)

        # optimization 
        optimizer.zero_grad()
        loss_iter.backward()    
        optimizer.step()

        # update weights and get the gradient norm

        weights = groupwise_weights(model, train_loader_large, loss_fn(reduction='none'), beta=temperature, device=device)

        if i % 10 == 0:
            # print("loss: ", loss_iter.data.item(), "acc: ", accuracy(yhat, labels), "grads: ", full_grd)
            print("loss: ", loss_iter.data.item(), "acc: ", accuracy(yhat, labels))

        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg, grad.avg, weights

def train_epoch_SVRG(model_k, model_snapshot, optimizer_k, optimizer_snapshot, 
                     train_loader, train_loader_large, start_weights, loss, acc,
                     grad, loss_fn, temperature=0.5):
    model_k.train()
    model_snapshot.train()

    
    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        yhat = model_snapshot(images)
        labels = labels.to(device)
        label_weights = torch.tensor([start_weights[label] for label in start_weights.keys()], dtype = torch.float32).to(device)
        
        snapshot_loss = loss_fn(weight=label_weights)(yhat, labels) / len(train_loader)
        snapshot_loss.backward()

    full_grd = torch.cat([param.grad.view(-1) for param in model_snapshot.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    grad.update(g)  

    # pass the current paramesters of optimizer_0 to optimizer_k 
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)
    weights = start_weights
    
    for i, (images, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print("Iteration: ", i)
        images = images.to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype = torch.float32).to(device)
        loss_iter = loss_fn(weight=label_weights)(yhat, labels)

        # optimization 
        optimizer_k.zero_grad()
        loss_iter.backward()    

        yhat2 = model_snapshot(images)
        loss2 = loss_fn(yhat2, labels)

        optimizer_snapshot.zero_grad()
        loss2.backward()

        optimizer_k.step(optimizer_snapshot.get_param_groups())

        # update weights
        # this is the average loss for each group
        
        weights = groupwise_weights(model_k, train_loader_large, loss_fn(reduction='none'), beta=temperature, device=device)

        # logging (using nohup to direct print to a file)
        if i % 10 == 0:
            print("Loss: ", loss_iter.data.item(), "Acc: ", accuracy(yhat, labels), "Grads: ", full_grd)
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, acc.avg, grad.avg, weights

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
    return loss_fns.get(loss_type)

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

def train_model(args, model, model_snapshot, optimizer, optimizer_snapshot, train_loader, train_loader_large, loss_fn, device):
    # loss_type = args.loss_type
    element_loss_fn = nn.NLLLoss(reduction='none') if loss_type == 'NLLLoss' else nn.CrossEntropyLoss(reduction='none')
    temperature = args.temperature
    start_weights = groupwise_weights(model, train_loader_large, element_loss_fn, beta=temperature, device=device)
    loss = AverageCalculator()
    acc = AverageCalculator()
    grad = AverageCalculator()

    train_loss_all, train_acc_all, weights_all, grads_all = [], [], [], []

    for epoch in range(args.n_epoch):
        t0 = time.time()

        if args.optimizer == "SGD":
            train_loss, train_acc, grads, new_weights = train_epoch_SGD(
                model, optimizer, train_loader, train_loader_large, start_weights, loss, acc, grad, loss_fn, temperature
            )
        elif args.optimizer == "SVRG":
            train_loss, train_acc, grads, new_weights = train_epoch_SVRG(
                model, model_snapshot, optimizer, optimizer_snapshot, train_loader, train_loader_large, start_weights, loss, acc, grad, loss_fn, temperature
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

    train_model(args, model, model_snapshot, optimizer, optimizer_snapshot, train_loader, train_loader_large, loss_fn, device)
            