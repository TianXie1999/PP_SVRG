import torch 




class CreditMLP(torch.nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
        super(CreditMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def get_credit_mlp(input_dim=10, hidden_dim=32, output_dim=2):
    return CreditMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)