import torch 
def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(5)
        device = 'cuda'
    print(f"Using device: {device}")
    return device