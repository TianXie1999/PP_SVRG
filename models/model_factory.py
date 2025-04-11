from .cifar100_models import CIFAR100_ConvNet, get_cifar100_ResNet18
from .cifar10_models import CIFAR10_ConvNet, get_cifar10_ResNet18
from .mnist_models import MNIST_one_layer, MNIST_two_layers, MNIST_ConvNet
from .credit_mlp import get_credit_mlp

model_dict = {
        "mnist_one_layer": MNIST_one_layer,
        "mnist_two_layers": MNIST_two_layers,
        "mnist_convnet": MNIST_ConvNet,
        "cifar10_convnet": CIFAR10_ConvNet,
        "cifar10_resnet18": get_cifar10_ResNet18,
        "cifar100_resnet18": get_cifar100_ResNet18,
        "credit_mlp": get_credit_mlp,
    }

def initialize_model(args, device):
    dataset = args.dataset
    model_name = args.nn_model.lower()
    if not model_name.startswith(dataset.lower()):
        model_name = f"{dataset.lower()}_{model_name}"
    print(f"Using model: {model_name}")
    NN_model = model_dict.get(model_name)
    if NN_model is None:
        raise ValueError("Unknown model")
    
    model = NN_model().to(device)
    model_snapshot = NN_model().to(device) if args.optimizer == 'SVRG' else None
    return model, model_snapshot