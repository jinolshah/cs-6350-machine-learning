import torch
from load_data import load_data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_X, train_y, test_X, test_y = load_data()

train_data = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32))

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, depth, width, activation):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_layer_size = input_size
        for _ in range(depth):
            layers.append(nn.Linear(prev_layer_size, width))
            if activation == 'tanh':
                nn.init.xavier_uniform_(layers[-1].weight)
            elif activation == 'relu':
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())
            prev_layer_size = width
        layers.append(nn.Linear(prev_layer_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels.unsqueeze(1)).sum().item()
    accuracy = correct / total
    return accuracy

input_size = train_X.shape[1]
output_size = 1
batch_size = 32
learning_rate = 1e-3

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['tanh', 'relu']

for depth in depths:
    for width in widths:
        for activation in activations:
            model = NeuralNetwork(input_size, output_size, depth, width, activation)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            print()
            print(f"Training with depth={depth}, width={width}, activation={activation}")
            train_model(model, train_loader, criterion, optimizer)
            train_accuracy = evaluate_model(model, train_loader)
            test_accuracy = evaluate_model(model, test_loader)
            print(f"Train error: {1 - train_accuracy}")
            print(f"Test error: {1 - test_accuracy}")
