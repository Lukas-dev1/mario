import torch
import torch.nn as nn

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MarioNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Mutation function for Evolutionary Algorithm
def mutate(model, mutation_rate=0.01):
    for param in model.parameters():
        param.data += mutation_rate * torch.randn_like(param)
