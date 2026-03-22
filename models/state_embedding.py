import numpy as np
import torch
import torch.nn as nn



class StateEmbedding(nn.Module):
    
    def __init__(self, height, width, d_model):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 1)
        
        self.residual = nn.Conv2d(32, 32, 1)
        
        self.conv3 = nn.Conv2d(32, 32, 1)
        self.conv4 = nn.Conv2d(32, 32, 1)
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(32 * height * width, d_model)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.residual(x)
        x = x + residual
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + residual
        
        x = self.flatten(x)
        x = self.fc(x)
        
        return x




if __name__ == "__main__":
    model = StateEmbedding(11, 6, 128)
    
    # Pass sample data
    x = torch.rand(16, 3, 64, 64)
    out = model(x)
    
    print(out.shape)
    # torch.Size([16, 512])