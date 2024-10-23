import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

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

# Create directories to save outputs
output_dir = "training_output"
os.makedirs(output_dir, exist_ok=True)

html_output_dir = "html_output"
os.makedirs(html_output_dir, exist_ok=True)

# Train the model
num_epochs = 1000
log_file_path = os.path.join(output_dir, "training_log.txt")

with open(log_file_path, "w") as log_file:
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Log the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            log_message = f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}\n"
            print(log_message)
            log_file.write(log_message)

            # Save the plot at this epoch
            plt.scatter(x.numpy(), y.numpy(), label="Original data", color="blue")
            predicted = model(x).detach()
            plt.plot(x.numpy(), predicted.numpy(), label="Fitted line", color="red")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title(f"Training Progress - Epoch {epoch + 1}")
            plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}.png"))
            plt.close()  # Close the plot to free memory

# Final plot
plt.scatter(x.numpy(), y.numpy(), label="Original data", color="blue")
predicted = model(x).detach()
plt.plot(x.numpy(), predicted.numpy(), label="Fitted line", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Final Fitted Line")
plt.savefig(os.path.join(output_dir, "final_plot.png"))
plt.close()

# Create an HTML file to display results in a separate directory
html_file_path = os.path.join(html_output_dir, "training_results.html")
with open(html_file_path, "w") as html_file:
    html_file.write("<html><head><title>Training Results</title></head><body>")
    html_file.write("<h1>Training Log</h1><pre>")
    with open(log_file_path, "r") as log_file:
        html_file.write(log_file.read())
    html_file.write("</pre><h2>Training Plots</h2>")

    # Embed the images
    for epoch in range(100, num_epochs + 1, 100):
        html_file.write(
            f'<img src="../{output_dir}/epoch_{epoch}.png" alt="Epoch {epoch} Plot"><br>'
        )

    # Embed the final plot
    html_file.write("<h3>Final Fitted Line</h3>")
    html_file.write(
        f'<img src="../{output_dir}/final_plot.png" alt="Final Fitted Line"><br>'
    )
    html_file.write("</body></html>")

print(f"Training log and plots saved to {html_file_path}")
