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


class NonLinearRegression(nn.Module):
    def __init__(self, hidden_size):
        super(NonLinearRegression, self).__init__()
        self.hidden = nn.Linear(1, hidden_size)  # 1 input feature, hidden_size hidden units
        self.relu = nn.ReLU()  # Non-linear activation function
        self.output = nn.Linear(hidden_size, 1)  # Output layer with 1 output feature

    def forward(self, x):
        x = self.hidden(x)  # First hidden layer
        x = self.relu(x)  # Apply non-linear activation
        x = self.output(x)  # Output layer
        return x


# Objective function for Optuna hyperparameter tuning
def objective(trial):
    # Hyperparameter search space
    hidden_size = trial.suggest_int('hidden_size', 5, 50)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)

    # Initialize the model, loss function, and optimizer with trial parameters
    model = NonLinearRegression(hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Set up TensorBoard writer for this trial
    writer = SummaryWriter(f'runs/optuna_trial_{trial.number}_lr_{lr}_hiddenSize_{hidden_size}')

    epochs = 500  # You can adjust the number of epochs
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(epochs):
        # Training phase
        model.train()
        y_train_pred = model(X_train)
        train_loss = criterion(y_train_pred, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        # Log training and validation loss to TensorBoard
        writer.add_scalar('Loss/train', train_loss.item(), epoch)
        writer.add_scalar('Loss/validation', val_loss.item(), epoch)

        # Save the entire model (including layers and weights) if it has the best validation loss so far
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model, f'best_model_trial_{trial.number}.pth')  # Save the entire model

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # Report validation loss to Optuna
        trial.report(val_loss.item(), epoch)

        # Early stopping condition: stop trial if validation loss is not improving
        if trial.should_prune():
            writer.close()  # Close TensorBoard writer
            raise optuna.exceptions.TrialPruned()

    writer.close()  # Close TensorBoard writer after training

    return best_val_loss


# Function to load the best model for predictions
def load_best_model(trial_number):
    # Load the entire model (with layers and weights)
    model = torch.load(f'best_model_trial_{trial_number}.pth')
    model.eval()  # Set the model to evaluation mode
    return model


# Run Optuna hyperparameter tuning
if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(),
        storage="sqlite:///db2.sqlite3",  # Specify the storage URL here.
        study_name="hello optuna"
    )
    study.optimize(objective, n_trials=10)  # You can adjust the number of trials

    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    # Get the trial number of the best trial
    best_trial_number = study.best_trial.number

    # Load the best model
    best_model = load_best_model(best_trial_number)

    # Example input for prediction
    new_data = torch.tensor([[2.0]])  # Example input data

    # Make predictions using the best model
    with torch.no_grad():
        prediction = best_model(new_data)
    print(f"Prediction for {new_data}: {prediction}")
