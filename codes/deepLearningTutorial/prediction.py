import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import optuna

# Generate data
X = torch.linspace(-5, 5, 100).reshape(-1, 1)  # Features
y = 3 * X**2 - X + 2 * torch.randn(X.size())  # Labels with noise

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from ex_2_optuna import NonLinearRegression

# Run Optuna hyperparameter tuning
if __name__ == '__main__':
    model = torch.load(f'best_model_trial_9.pth', weights_only=False)
    model.eval()  # Set the model to evaluation mode

    # Example input for prediction
    new_data = torch.tensor([[2.0],[8.]])  # Example input data

    # Make predictions using the best model
    with torch.no_grad():
        prediction = model(new_data)
    print(f"Prediction for {new_data}: {prediction}")
