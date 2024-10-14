import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
import numpy as np

import os

# Set TMPDIR to a shorter path
os.environ["TMPDIR"] = "C:/temp"

# Verify that the directory exists or create it
os.makedirs("C:/temp", exist_ok=True)

# Sample Data: y = 2x + 3 (with some noise)
np.random.seed(42)
x = np.linspace(-5, 5, 50)
y = 2 * x + 3 + np.random.normal(scale=1, size=x.shape)

# Convert data to PyTorch tensors
X = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Simple Linear Model: y = mx + c
class LinearModel(nn.Module):
    def __init__(self, config):
        super(LinearModel, self).__init__()
        # Define a simple neural network with one hidden layer
        self.hidden = nn.Linear(1, config['units'])  # 1 input feature, 10 hidden units
        self.relu = nn.ReLU()  # Non-linear activation function
        self.output = nn.Linear(config['units'], 1)  # Output layer with 1 output feature
    
    def forward(self, x):
        x = self.hidden(x)  # First hidden layer
        x = self.relu(x)  # Apply non-linear activation
        x = self.output(x)  # Output layer
        return x

# Training function used by Ray Tune
def train_model(config):
    model = LinearModel(config)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])  # Using Ray Tune's LR

    # Train for a few epochs
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()

    # Report loss to Ray Tune
    return {"loss":loss.item()}

# Ray Tune Hyperparameter Search Space
search_space = {"lr": tune.loguniform(1e-4, 1e-1), "units": tune.grid_search([4,8])}  # Explore learning rates

# Run Ray Tune
analysis = tune.run(
    train_model,
    config=search_space,
    metric="loss",
    mode="min",
    num_samples=10  # Run 10 trials with different learning rates
)

# Get the best trial
best_trial = analysis.get_best_trial("loss", "min", "all")
print(f"Best Trial Config: {best_trial.config}")
print(f"Best Loss: {best_trial.last_result['loss']}")
