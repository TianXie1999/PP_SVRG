import torch
from torch import nn 
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
from sgd import SGD_Simple
from svrg import SVRG_k, SVRG_Snapshot
from utils import MNIST_dataset, MNIST_dataset_sample, CIFAR10_dataset, MNIST_two_layers, MNIST_one_layer, MNIST_ConvNet, CIFAR10_ConvNet, AverageCalculator, accuracy

parser = argparse.ArgumentParser(description="Train SVRG/SGD on MNIST data.")
parser.add_argument('--optimizer', type=str, default="SVRG",
                    help="optimizer.")
parser.add_argument('--nn_model', type=str, default="MNIST_one_layer",
                    help="neural network model.")
parser.add_argument('--dataset', type=str, default="MNIST",
                    help="neural network model.")
parser.add_argument('--n_epoch', type=int, default=100,
                    help="number of training iterations.")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate.")
parser.add_argument('--batch_size', type=int, default=64,
                    help="batch size.")
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help="regularization strength.")
parser.add_argument('--exp_name', type=str, default="",
                    help="name of the experiment.")
parser.add_argument('--print_every', type=int, default=1,
                    help="how often to print the loss.")
parser.add_argument('--ratio', type=float, default=1.0,
                    help="how much of the data to use.")
parser.add_argument('--temperature', type=float, default=0.5,
                    help="temperature for softmax.")

OUTPUT_DIR = "outputs"

device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(2)
    device = 'cuda'
print("Using device: {}".format(device))

def train_epoch_SGD(model, optimizer, train_loader, train_loader_large, start_weights, loss, acc, grad, flatten_img=True, loss_type='NLLLoss', temperature=0.5):
    model.train()
    # calculate the mean gradient
    optimizer.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model(images)
        labels = labels.to(device)
        label_weights = torch.tensor([start_weights[label] for label in start_weights.keys()], dtype=torch.float32).to(device)
        if loss_type == 'NLLLoss':
            loss_fn = nn.NLLLoss(weight=label_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=label_weights)
        loss_iter = loss_fn(yhat, labels) / len(train_loader)
        loss_iter.backward()
    
    full_grd = torch.cat([param.grad.view(-1) for param in model.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    grad.update(g)

    weights = start_weights
    i = 0
    for images, labels in train_loader:
        i += 1
        if i % 10 == 0:
            print("Iteration: ", i)
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model(images)
        labels = labels.to(device)
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
        if loss_type == 'NLLLoss':
            loss_fn = nn.NLLLoss(weight=label_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=label_weights)
        loss_iter = loss_fn(yhat, labels)

        # optimization 
        optimizer.zero_grad()
        loss_iter.backward()    
        optimizer.step()

        # update weights and get the gradient norm
        if loss_type == 'NLLLoss':
            loss_fn = nn.NLLLoss(reduction='none')
        else:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        weights = groupwise_weights(model, train_loader_large, loss_fn, flatten_img=flatten_img, beta=temperature)

        if i % 10 == 0:
            print("loss: ", loss_iter.data.item(), "acc: ", accuracy(yhat, labels), "grads: ", full_grd)

        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg, grad.avg, weights

def train_epoch_SVRG(model_k, model_snapshot, optimizer_k, optimizer_snapshot, train_loader, train_loader_large, start_weights, loss, acc, grad, flatten_img=True, loss_type='NLLLoss', temperature=0.5):
    model_k.train()
    model_snapshot.train()

    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_snapshot(images)
        labels = labels.to(device)
        label_weights = torch.tensor([start_weights[label] for label in start_weights.keys()], dtype = torch.float32).to(device)
        if loss_type == 'NLLLoss':
            loss_fn = nn.NLLLoss(weight=label_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=label_weights)
        snapshot_loss = loss_fn(yhat, labels) / len(train_loader)
        snapshot_loss.backward()

    full_grd = torch.cat([param.grad.view(-1) for param in model_snapshot.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    grad.update(g)  

    # pass the current paramesters of optimizer_0 to optimizer_k 
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)
    weights = start_weights
    i = 0
    for images, labels in train_loader:
        i += 1
        if i % 10 == 0:
            print("Iteration: ", i)
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_k(images)
        labels = labels.to(device)
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype = torch.float32).to(device)
        if loss_type == 'NLLLoss':
            loss_fn = nn.NLLLoss(weight=label_weights)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=label_weights)
        loss_iter = loss_fn(yhat, labels)

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
        if loss_type == 'NLLLoss':
            element_loss_fn = nn.NLLLoss(reduction='none')
        else:
            element_loss_fn = nn.CrossEntropyLoss(reduction='none')
        weights = groupwise_weights(model_k, train_loader_large, element_loss_fn, flatten_img=flatten_img, beta=temperature)

        # logging (using nohup to direct print to a file)
        if i % 10 == 0:
            print("Loss: ", loss_iter.data.item(), "Acc: ", accuracy(yhat, labels), "Grads: ", full_grd)
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, acc.avg, grad.avg, weights


def groupwise_weights(model_k, train_loader_large, loss_fn, flatten_img=True, beta=0.5):
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
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_k(images)
        labels = labels.to(device)
        # loss for each sample without averaging
        loss_iter = loss_fn(yhat, labels)
        loss_iter = loss_iter.view(-1)  # Ensure loss_iter is a 1D tensor
        unique_labels = torch.unique(labels)
        for i in range(len(unique_labels)):
            label = unique_labels[i].item()
            group_loss[label] = loss_iter[labels == label].mean()  
    # calculate the weights
    group_weights = {}
    for label in group_loss.keys():
        group_weights[label] = np.exp(-beta * group_loss[label].item())
    total = sum(group_weights.values())
    for label in group_weights.keys():
        group_weights[label] /= total
        group_weights[label] *= len(group_weights)  # scale to the number of groups
    # print("Group weights: ", group_weights)
    return group_weights



if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    if not args.optimizer in ['SGD', 'SVRG']:
        raise ValueError("--optimizer must be 'SGD' or 'SVRG'.")
    print(args_dict)

    # load the data
    if args.dataset == "MNIST":
        if args.ratio < 1:
            train_set, val_set = MNIST_dataset_sample(p=args.ratio)
        else:   
            train_set, val_set = MNIST_dataset()
        if args.nn_model == "MNIST_ConvNet":
            flatten_img = False
        else:
            flatten_img = True

    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset() 
        flatten_img = False

    else:
        raise ValueError("Unknown dataset")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    train_loader_large = DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    loss_type = 'NLLLoss'
    if args.nn_model == "MNIST_one_layer":
        NN_model = MNIST_one_layer  # function name 
    elif args.nn_model == "MNIST_two_layers":
        NN_model = MNIST_two_layers
    elif args.nn_model == "MNIST_ConvNet":
        NN_model = MNIST_ConvNet
        loss_type = 'CrossEntropyLoss'
    elif args.nn_model == "CIFAR10_convnet":
        NN_model = CIFAR10_ConvNet
        loss_type = 'CrossEntropyLoss'
    else:
        raise ValueError("Unknown nn_model.")

    model = NN_model().to(device)
    if args.optimizer == 'SVRG':
        model_snapshot = NN_model().to(device)

    lr = args.lr  # learning rate
    n_epoch = args.n_epoch  # the number of epochs

    # the optimizer 
    if args.optimizer == "SGD":
        optimizer = SGD_Simple(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SVRG":
        optimizer = SVRG_k(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())


    # output folder 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = timestamp + "_" + args.optimizer + "_" + args.nn_model + "_Temperature" + str(args.temperature) + "_lr" + str(lr) 
    if args.exp_name != "":
        model_name = args.exp_name + '_' + model_name
    log_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    # store training stats
    train_loss_all, val_loss_all, weights_all, grads_all = [], [], [], []
    train_acc_all, val_acc_all = [], []
    if loss_type == 'NLLLoss':
        element_loss_fn = nn.NLLLoss(reduction='none')
    else:
        element_loss_fn = nn.CrossEntropyLoss(reduction='none')
    temperature = args.temperature
    start_weights = groupwise_weights(model, train_loader_large, element_loss_fn, flatten_img=flatten_img, beta=temperature)
    loss = AverageCalculator()
    acc = AverageCalculator()
    grad = AverageCalculator()

    for epoch in range(n_epoch):
        t0 = time.time()

        # training 
        if args.optimizer == "SGD":
            train_loss, train_acc, grads, new_weights = train_epoch_SGD(model, optimizer, train_loader, train_loader_large, start_weights, loss, acc, grad, flatten_img=flatten_img, loss_type = loss_type, temperature=temperature)
        elif args.optimizer == "SVRG":
            train_loss, train_acc, grads, new_weights = train_epoch_SVRG(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, train_loader_large, start_weights, loss, acc, grad, flatten_img=flatten_img, loss_type = loss_type, temperature=temperature)
  
        train_loss_all.append(train_loss)  # averaged loss for the current epoch 
        train_acc_all.append(train_acc)
        weights_all.append(new_weights)
        grads_all.append(grads)
        
        fmt_str = "epoch: {}, train loss: {:.4f}, train acc: {:.4f}, train gradient: {:.4f}, train weights: {:.4f}, time: {:.2f}"

        if epoch % args.print_every == 0:
            print(f"Epoch {epoch} / {n_epoch}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")
        
        start_weights = new_weights

        # save data and plot 
        if (epoch + 1) % 1 == 0:
            np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all), weights=np.array(weights_all), grads = np.array(grads_all))

    # done
    open(os.path.join(log_dir, 'done'), 'a').close()
            