import os 
import argparse




LR_RANGE = [0.001]



def SGD_lr_search():
    arg_template = [
        'python', 'main.py',
        '--optimizer', 'SGD',
        '--n_epoch', '800',
        '--dataset', 'CIFAR10',
        '--nn_model', 'CIFAR10_convnet',
        '--device', 'cuda:5',
        '--batch_size', '5',
        '--temperature', '5',
        '--ratio', '0.1',
        '--log',
        '--lr', 
    ]
    for lr in LR_RANGE:
        cmd = ' '.join(arg_template + [str(lr)])
        print(cmd)
        os.system(cmd)
        
        
def SVRG_lr_search():
    arg_template = [
        'python', 'main.py',
        '--optimizer', 'SVRG',
        '--dataset', 'CIFAR10',
        '--nn_model', 'CIFAR10_convnet',
        '--device', 'cuda:5',
        '--batch_size', '5',
        '--temperature', '5',
        '--ratio', '0.1',
        '--log',
        '--lr', 
    ]
    for lr in LR_RANGE:
        cmd = ' '.join(arg_template + [str(lr)])
        print(cmd)
        os.system(cmd)
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--optim', type=str, default='SGD')
    if args.parse_args().optim == 'SGD':
        SGD_lr_search()
    elif args.parse_args().optim == 'SVRG':
        SVRG_lr_search()
        
        