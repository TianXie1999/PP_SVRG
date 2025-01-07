import torch 
def get_device(device):
    
    if torch.cuda.is_available():
        if device == 'cuda':
            torch.cuda.set_device(0)
            device = 'cuda'
        elif device.startswith('cuda'):
            try:
                torch.cuda.set_device(device)
                
            except Exception:
                print(f"Invalid device: {device}. Using default device: {device}")
        elif int(device):
            try :
                torch.cuda.set_device(int(device))
                device = f'cuda:{device}'
            except Exception:
                print(f"Invalid device: {device}. Using default device: {device}")
    else:
        device = 'cpu'

    print(f"Using device: {device}")
    return device