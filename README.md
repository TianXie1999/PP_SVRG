# Nonconvex PP SVRG (PyTorch)
Implementation of **stochastic variance reduction gradient descent (SVRG)** for optimizing non-convex neural network functions in PyTorch but considering a performative prediction setting

# How to run

template for running with SGD
```bash
python main.py --optimizer SGD --dataset MNIST --nn_model one_layer --lr 0.001 --device 0 --log
```

template for running with SVRG
```bash
python main.py --optimizer SVRG --lr 0.0001 --log # runnning SVRG on mnist 
python main.py --optimizer SVRG --dataset CIFAR100 --nn_model resnet18 --lr 0.001 --log # running 

```

Up to now, the support args
- --optimizer: ['SGD','SVRG']. The optimizer to be used.
- --nn_model:
- --dataset: ['MNIST', 'CIFAR10', 'CIFAR100']. The dataset to be used.
- --n_epochs: Int. Number of training iterations.
- --lr: float. Learning rate.
- --batch_size: Int. Batch size.
- --log: store_true. Whether log the results in a log file. Default is False.
- --weight_decay: float. Weight decay for the optimizer. Default is 0.
- --print_every: Int. Print the results every n iterations. Default is 1.
- --ratio: float. The ratio of the dataset to be used. Default is 1.0.
- --temperture: float. The temperature for the performative prediction setting. Default is 1.0.
- --loss_type: ['cross_entropy', 'NLL']. The loss function to be used. Default is 'cross_entropy'.
- --output_dir: str. The output directory for the log file. Default is 'logs'.
- --device: ['cuda', 'cpu']. The device to be used. Default is 'cuda'.
