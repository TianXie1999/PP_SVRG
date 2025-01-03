import torch 
import torch.nn as nn
from .utils import calculate_full_gradient, calculate_loss, update_weights, log_metrics



def train_one_epoch(model, optimizer, train_loader, train_loader_large, start_weights,
                    loss, acc, grad, loss_fn, model_snapshot=None, optimizer_snapshot=None,
                    temperature=0.5, optimize='SGD', device='cpu'):
    g = calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer,
                                device)
    grad.update(g)

    weights = start_weights
    print("total iterations:", len(train_loader))
    for i, (images, labels) in enumerate(train_loader):
        if i % 10 == 0:
            print("Iteration: ", i)

        # Calculate loss and perform optimization
        loss_iter, yhat = calculate_loss(model, images, labels, weights, loss_fn, device)
        optimizer.zero_grad()
        loss_iter.backward()    
        optimizer.step()

        # Update weights
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
        
        weights = update_weights(model, train_loader_large, loss_fn(weight=label_weights), temperature, device)

        # Log metrics
        
        log_metrics(loss_iter, yhat, labels, loss, acc)

    return loss.avg, acc.avg, grad.avg, weights

def train_epoch_SVRG(model_k, model_snapshot, optimizer_k, optimizer_snapshot, 
                     train_loader, train_loader_large, start_weights, loss, acc,
                     grad, loss_fn, temperature=0.5, device='cpu'):
    model_k.train()
    g = calculate_full_gradient(model_snapshot, train_loader, start_weights, loss_fn, 
                                optimizer_snapshot, device)
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
        
        
        weights = update_weights(model_k, train_loader_large, loss_fn(reduction='none')
                                 , beta=temperature, device=device)
        # logging (using nohup to direct print to a file)
        log_metrics(loss_iter, yhat, labels, loss, acc)
    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, acc.avg, grad.avg, weights