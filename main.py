import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate synthetic data
x = torch.randn(100, 1) * 10  # 100 data points
y = 2 * x + 3 + torch.randn(100, 1)  # Linear relationship with some noise


# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x)


# Instantiate the model, define loss function and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Stochastic Gradient Descent

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

    if (epoch + 1) % 100 == 0:  # Print every 100 epochs
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Plot the results
predicted = model(x).detach()  # Get predictions

plt.scatter(x.numpy(), y.numpy(), label="Original data", color="blue")
plt.plot(x.numpy(), predicted.numpy(), label="Fitted line", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Simple Linear Regression with PyTorch")
plt.show()
