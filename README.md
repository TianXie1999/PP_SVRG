# Nonconvex PP SVRG (PyTorch)
Implementation of **stochastic variance reduction gradient descent (SVRG)** for optimizing non-convex neural network functions in PyTorch but considering a performative prediction setting

# How to run

template for running with SGD
```bash
python main.py --optimizer SGD --dataset MNIST --nn_model one_layer --lr 0.003 --device 0 --log --temperature 5 --ratio 0.05
python main.py --optimizer SVRG --dataset MNIST --nn_model one_layer --lr 0.003 --device 0 --log --temperature 5 --ratio 0.05 --batch_size 32
```

template for running with SVRG
```bash
python main.py --optimizer SVRG --lr 0.0001 --log # runnning SVRG
python main.py --optimizer SVRG --dataset CIFAR100 --nn_model resnet18 --lr 0.001 --log --device 1 # running SVRG on CIFAR100 with ResNet18 on GPU 1
python main.py --optimizer SVRG --dataset CIFAR100 --nn_model CIFAR100_convnet --lr 0.001 --log --device cuda:1 --batch_size  # running SVRG on CIFAR100 with LeNet on GPU 1
python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.008 --log --device cuda:5 --batch_size 256 --temperature 5
python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.005 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.005 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1
python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.001 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.001 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1
python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.01 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.01 --log --device cuda:5 --batch_size 5 --temperature 5 --ratio 0.1

python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 10 --temperature 5 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 10 --temperature 5 --ratio 0.1

python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 100 --temperature 10 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 100 --temperature 10 --ratio 0.1

python main.py --optimizer SVRG --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 100 --temperature 20 --ratio 0.1
python main.py --optimizer SGD --dataset CIFAR10 --nn_model CIFAR10_convnet --lr 0.05 --log --device cuda:5 --batch_size 100 --temperature 20 --ratio 0.1

WANDB_AGENT_DISABLE_FLAPPING=true wandb agent --count 20 your_entity_name/svrg_sgd_cifar/abc123456 --python sweep.py
```
python main.py --optimizer SVRG --dataset MNIST --nn_model one_layer --lr 0.003 --device 0 --log --temperature 50 --ratio 0.05 --batch_size 100

## Creidt dataset 
If you want to run with the credit dataset, please use the following command
```
python credit.py --dataset credit --nn_model mlp --ratio 1 --log --optimizer SVRG
```
The dataset is from ![kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)


Up to now, the support args
- --optimizer: ['SGD','SVRG']. The optimizer to be used.
- --nn_model:
- --dataset: ['MNIST', 'CIFAR10', 'CIFAR100', 'credit']. The dataset to be used.
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
