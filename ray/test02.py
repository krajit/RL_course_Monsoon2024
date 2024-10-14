import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from ray import tune
import ray

# Generate data
X = torch.linspace(-5, 5, 100).reshape(-1, 1)  # Features
y = 3 * X**2 - X + 2 * torch.randn(X.size())  # Labels with noise

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the NonLinearRegression model
class NonLinearRegression(nn.Module):
    def __init__(self, hidden_units):
        super(NonLinearRegression, self).__init__()
        self.hidden = nn.Linear(1, hidden_units)  # Input layer to hidden
        self.relu = nn.ReLU()  # Non-linear activation function
        self.output = nn.Linear(hidden_units, 1)  # Hidden to output layer
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
hidden_units = tune.choice([8, 16, 32])  # Example of hyperparameter tuning
learning_rate = tune.loguniform(1e-4, 1e-1)  # Learning rate range

def train_model(hidden_units, learning_rate):
    model = NonLinearRegression(hidden_units).to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to tensors and send to device
    X_train_tensor = X_train.to(device)
    y_train_tensor = y_train.to(device)

    for epoch in range(1000):  # Number of epochs
        model.train()
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # Validate the model
    model.eval()
    with torch.no_grad():
        X_val_tensor = X_val.to(device)
        y_val_tensor = y_val.to(device)
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    return val_loss.item()

# Initialize Ray
ray.init()

import os
storage_dir = os.path.abspath("./run4")
# Hyperparameter tuning with Ray Tune
analysis = tune.run(
    train_model,
    config={
        "hidden_units": tune.choice([8, 16, 32]),  # Example of hyperparameter tuning
        "learning_rate": tune.loguniform(1e-4, 1e-1)  # Learning rate range
    },
    num_samples=10,  # Number of trials
    storage_path = storage_dir
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="val_loss", mode="min")
print("Best config: ", best_config)

# Optional: Plotting results
model = NonLinearRegression(best_config['hidden_units']).to(device)
# Train the model using the best hyperparameters
train_model(best_config['hidden_units'], best_config['learning_rate'])

# Generate predictions for plotting
with torch.no_grad():
    model.eval()
    y_pred = model(X.to(device)).cpu()

# Plotting
plt.scatter(X.cpu(), y.cpu(), label='Data', alpha=0.5)
plt.plot(X.cpu(), y_pred, color='red', label='Predictions', linewidth=2)
plt.legend()
plt.title('Non-Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
