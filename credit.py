import os 
os.environ['WANDB_SILENT'] = 'true'
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader

from optim import initialize_optimizer
from dataset import load_dataset
from models import initialize_model
from utils import get_device, get_args, get_loss_fn, train_credit_model, setup_output_directory, log_to_file
import hydra
import wandb




if __name__ == "__main__":
    args = get_args()
    
    log_dir = setup_output_directory(args)
    
    device = get_device(args.device)
    
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
    @log_to_file(os.path.join(log_dir, 'training.log'), log=args.log)
    def decorated_train_model(*args, **kwargs):
        return train_credit_model(*args, **kwargs)
    
    decorated_train_model(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, 
                train_loader_large, loss_fn, log_dir, n_epochs=args.n_epoch, optimize=args.optimizer,
                temperature = args.temperature, print_interval=args.print_every, device=device,
                log=args.log, use_wandb=args.wandb
            )
            