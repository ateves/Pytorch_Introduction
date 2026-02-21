import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print(f"PyTorch Version: {torch.__version__}")

# 1. Define the model class (simple linear regression: y = 2x + 1)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # One input feature, one output feature
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 2. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Prepare training data (convert numpy arrays to PyTorch tensors)
# Expected output is roughly 2*x + 1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

inputs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
targets = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

# 4. Train the model
print("\nStarting PyTorch training...")
for epoch in range(500):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad() # Clear gradients
    loss.backward()      # Compute gradients
    optimizer.step()     # Update weights

print("PyTorch training complete.")

# 5. Make a prediction
model.eval() # Set model to evaluation mode
with torch.no_grad():
    prediction_x = torch.tensor([10.0], dtype=torch.float32).view(-1, 1)
    prediction_y = model(prediction_x).item()
    print(f"\nPrediction for x = 10.0: {prediction_y:.2f} (Expected: ~21.0)")
