from .cifar100_models import CIFAR100_ConvNet, get_ResNet18
from .cifar10_models import CIFAR10_ConvNet
from .mnist_models import MNIST_one_layer, MNIST_two_layers, MNIST_ConvNet
def initialize_model(args, device):
    model_dict = {
        "MNIST_one_layer": MNIST_one_layer,
        "MNIST_two_layers": MNIST_two_layers,
        "MNIST_ConvNet": MNIST_ConvNet,
        "CIFAR10_convnet": CIFAR10_ConvNet,
        "CIFAR100_resnet18": get_ResNet18,
    }
    NN_model = model_dict.get(args.nn_model)
    if NN_model is None:
        raise ValueError("Unknown model")
    model = NN_model().to(device)
    model_snapshot = NN_model().to(device) if args.optimizer == 'SVRG' else None
    return model, model_snapshot