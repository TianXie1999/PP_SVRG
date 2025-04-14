import os
import time
import torch 

import pandas as pds
import numpy as np
import wandb

from .utils import calculate_full_gradient, calculate_loss, update_weights, log_metrics, update_dataset
from .log import AverageCalculator, log_to_file
from .metric import accuracy


def train_one_epoch(model, optimizer, train_loader, train_loader_large, start_weights,
                    metric, loss_fn, model_snapshot=None, optimizer_snapshot=None,
                    temperature=0.5, optimize='SGD', device='cpu', update_weight=True):

    if optimize == 'SVRG':
        g = calculate_full_gradient(model_snapshot, train_loader, start_weights, loss_fn, 
                                    optimizer_snapshot, device)
    elif optimize == 'SGD':
        g = calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer,
                                    device)

    metric['grad'].update(g)
    
    if optimize == 'SVRG':
        u = optimizer_snapshot.get_param_groups()
        optimizer.set_u(u)
        
    
    weights = start_weights
    print("total iterations:", len(train_loader))

    for i, (images, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Weights: ", weights)

        # Calculate loss and perform optimization
        loss_iter, yhat = calculate_loss(model, images, labels, weights, loss_fn, device)
        # if loss_iter.mean() > 100:
        #     print(yhat)   
            # print(weights)
        optimizer.zero_grad()
        loss_iter.backward()    
        
        if optimize == 'SVRG':
            loss2, _ = calculate_loss(model_snapshot, images, labels, weights, loss_fn, device)
            optimizer_snapshot.zero_grad()         
            loss2.backward()
            optimizer.step(optimizer_snapshot.get_param_groups())
        else:
            optimizer.step()  
        # Update weights
        # label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
        if update_weight:
            with torch.no_grad():
                weights = update_weights(model, train_loader_large, loss_fn, 
                                 temperature, device)
        # weights = update_weights(model, train_loader_large, loss_fn, 
                                #  temperature, device)
        else:
            update_dataset(3, train_loader, model, device, alpha=0.1)
        # Log metrics
        acc = accuracy(yhat.cpu(), labels)
        log_metrics(loss_iter, acc, metric, i)
    
    if optimize == 'SVRG':
        
        optimizer_snapshot.set_param_groups(optimizer.get_param_groups())
        
        
    return metric['loss'].avg, metric['acc'].avg, metric['grad'].avg, weights

def train_model(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, 
                train_loader_large, loss_fn, log_dir, n_epochs, optimize,
                print_interval, temperature, device, log, use_wandb):
    
    start_weights = update_weights(model, train_loader_large, loss_fn, beta=temperature, device=device)
    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
        'grad': AverageCalculator(),
    }

    columns = ['epoch', 'train_loss', 'train_acc', 'weights', 'grads']
    df = pds.DataFrame(columns=columns)

    for epoch in range(n_epochs):
        t0 = time.time()


        train_loss, train_acc, grads, new_weights = train_one_epoch(
                model, optimizer, train_loader, train_loader_large, start_weights, metrics, 
                loss_fn, model_snapshot, optimizer_snapshot, temperature, 
                optimize=optimize, device=device
            )
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'grads': grads,
                'weights': new_weights
            })

        # for metric in metrics.values():
        #     metric.reset()

        new_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'weights': new_weights,
            'grads': grads
        }
        # use concat
        df = pds.concat([df, pds.DataFrame(new_row, index=[0])], ignore_index=True)
        print(df)

        if epoch % print_interval == 0:
            print(f"Epoch {epoch} / {n_epochs}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")
            
            

        start_weights = new_weights

        if (epoch + 1) % 1 == 0 and log:
            df.to_csv(os.path.join(log_dir, 'train_stats.csv'))
    if log:
        open(os.path.join(log_dir, 'done'), 'a').close()
        
        
def train_credit_model(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, 
                train_loader_large, loss_fn, log_dir, n_epochs, optimize,
                print_interval, temperature, device, log, use_wandb):
    
    update_dataset(3, train_loader, model, device, alpha=0.1)
    
    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
        'grad': AverageCalculator(),
    }

    columns = ['epoch', 'train_loss', 'train_acc', 'weights', 'grads']
    df = pds.DataFrame(columns=columns)

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, train_acc, grads, new_weights = train_one_epoch(
                model, optimizer, train_loader, train_loader_large, None, metrics, 
                loss_fn, model_snapshot, optimizer_snapshot, temperature, 
                optimize=optimize, device=device, update_weight=False
            )
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'grads': grads,
                'weights': new_weights
            })

        # for metric in metrics.values():
        #     metric.reset()

        new_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'weights': new_weights,
            'grads': grads
        }
        # use concat
        df = pds.concat([df, pds.DataFrame(new_row, index=[0])], ignore_index=True)
        print(df)

        if epoch % print_interval == 0:
            print(f"Epoch {epoch} / {n_epochs}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")
            
            


        if (epoch + 1) % 1 == 0 and log:
            df.to_csv(os.path.join(log_dir, 'train_stats.csv'))
    if log:
        open(os.path.join(log_dir, 'done'), 'a').close()