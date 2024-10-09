import torch

# Define the function
def myFun(x, y):
    return x**2 + y**2

# Define x and y as tensors with requires_grad=True
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# Call the function
f = myFun(x, y)

# Compute the gradient
f.backward()

# Print the gradients of x and y
print(f"Gradient of f with respect to x: {x.grad}")
print(f"Gradient of f with respect to y: {y.grad}")
