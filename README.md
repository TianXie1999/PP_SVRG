# Nonconvex PP SVRG (PyTorch)
Implementation of **stochastic variance reduction gradient descent (SVRG)** for optimizing non-convex neural network functions in PyTorch but considering a performative prediction setting

# How to run

template for running with SGD
```bash
python main.py --optimizer SGD --lr 0.001 --log
```

template for running with SVRG
```bash
python main.py --optimizer SVRG --lr 0.0001 --log # runnning SVRG on mnist 
python main.py --optimizer SVRG --dataset CIFAR100 --nn_model resnet18 --lr 0.001 --log # running 

```

Up to now, the support args
- --optimizer: ['SGD','SVRG'] The optimizer to be used.
- --