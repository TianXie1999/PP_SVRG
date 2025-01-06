import torch.nn as nn
def get_loss_fn(loss_type):
    loss_fns = {
        'NLLLoss': nn.NLLLoss,
        'CrossEntropyLoss': nn.CrossEntropyLoss
    }
    return loss_fns.get(loss_type, nn.CrossEntropyLoss)