import argparse


def get_args():
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
    parser.add_argument('--loss_type', type=str, default="cross_entropy", 
                        help="loss function.")
    return parser.parse_args()