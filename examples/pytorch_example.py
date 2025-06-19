# PyTorch neural network example
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_model():
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()
    
    # Create model
    model = SimpleNN(10, 20, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    return model

def predict(model, X):
    with torch.no_grad():
        predictions = model(X)
    return predictions

if __name__ == "__main__":
    model = train_model()
    test_input = torch.randn(5, 10)
    predictions = predict(model, test_input)
    print("Predictions:", predictions)
