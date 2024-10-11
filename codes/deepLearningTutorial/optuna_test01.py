import torch
import optuna

# Points to fit the line
x_data = torch.tensor([0.0, 1.0, -2.0, 6.0])
y_data = torch.tensor([0.0, 2.5, 0.0, 4.0])

# Define the loss function
def loss_fn(m, c, x_data, y_data):
    y_pred = m * x_data + c
    return torch.mean((y_pred - y_data) ** 2)

# Define the optimization function for Optuna
def objective(trial):
    # Initialize m and c with requires_grad=True for autograd
    m = torch.tensor(trial.suggest_uniform('m', -10, 10), requires_grad=True)
    c = torch.tensor(trial.suggest_uniform('c', -10, 10), requires_grad=True)
    
    # Suggest learning rate (alpha) from Optuna
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    
    # Gradient Descent loop
    for _ in range(100):
        # Compute the loss
        loss = loss_fn(m, c, x_data, y_data)
        
        # Compute gradients
        loss.backward()
        
        # Update m and c using gradient descent
        with torch.no_grad():
            m -= alpha * m.grad
            c -= alpha * c.grad
            
            # Reset gradients to zero
            m.grad.zero_()
            c.grad.zero_()
    
    return loss.item()  # Return the final loss after 100 iterations


# Create an Optuna study
study = optuna.create_study(
    direction='minimize',
            storage="sqlite:///db3.sqlite3",  # Specify the storage URL here.
            study_name="TestOptuna"
    )

# Run the optimization
study.optimize(objective,
               n_trials=50,
               )

# Print the best parameters
best_params = study.best_params
print(f"Best parameters: m={best_params['m']}, c={best_params['c']}, alpha={best_params['alpha']}")
