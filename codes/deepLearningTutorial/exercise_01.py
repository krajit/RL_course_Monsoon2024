import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # TensorBoard writer

# Generate data
X = torch.linspace(-5, 5, 100).reshape(-1, 1)  # Features
y = 3 * X  + 2*torch.randn(X.size())  # Labels with noise



class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Define a simple neural network with one hidden layer
        self.hidden = nn.Linear(1, 10)  # 1 input feature, 10 hidden units
        self.relu = nn.Tanh()  # Non-linear activation function
        self.output = nn.Linear(10, 1)  # Output layer with 1 output feature
    
    def forward(self, x):
        x = self.hidden(x)  # First hidden layer
        x = self.relu(x)  # Apply non-linear activation
        x = self.output(x)  # Output layer
        return x

# Initialize model, loss function, and optimizer
model = RegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Set up TensorBoard writer
writer = SummaryWriter('runs/non_linear_regression_experiment')

# Visualization function
def plot_predictions(epoch, model, X, y):
    plt.clf()  # Clear the previous plot
    model.eval()
    predicted = model(X).detach().numpy()
    plt.scatter(X.numpy(), y.numpy(), label='Original data')
    plt.plot(X.numpy(), predicted, label=f'Prediction at epoch {epoch}', color='green')
    plt.legend()
    plt.title(f'Prediction vs Actual Data at Epoch {epoch}')
    plt.pause(0.1)  # Pause to update the plot

# Prepare to visualize in real-time
plt.ion()  # Enable interactive mode

# Train the non-linear model and visualize predictions at intervals
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    
     # Log the loss to TensorBoard
    writer.add_scalar('Training Loss', loss.item(), epoch)
    
    # Visualize every 100 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        plot_predictions(epoch+1, model, X, y)

plt.ioff()  # Disable interactive mode after training
plt.show()  # Show the final plot
