
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import syft as sy  # Untuk Federated Learning

# ========================
# 1. Linear Regression
# ========================
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
lin_reg = LinearRegression().fit(X, y)
print("Linear Regression Coef:", lin_reg.coef_)

# ========================
# 2. Two-Layer Neural Network
# ========================
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TwoLayerNN(1, 5, 1)

# Tes output awal sebelum training
sample_input = torch.tensor([[3.0]])
initial_output = model(sample_input)
print("NN Output (before training):", initial_output.item())

# ========================
# 3. Optimisation & Training
# ========================
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Data untuk training
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# Training loop
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

print("Loss after training:", loss.item())
trained_output = model(sample_input)
print("NN Output (after training):", trained_output.item())

# ========================
# 4. AI in Privacy - Federated Learning
# ========================
from syft.frameworks.torch import TorchHook
from syft.workers.virtual import VirtualWorker

hook = TorchHook(torch)
worker1 = VirtualWorker(hook, id="worker1")
worker2 = VirtualWorker(hook, id="worker2")

# Simulasi kirim data ke worker1
data = torch.tensor([[1.0], [2.0]]).send(worker1)
label = torch.tensor([[2.0], [4.0]]).send(worker1)

# Kirim model ke worker1 dan hitung output
model.send(worker1)
remote_output = model(data)
print("Remote output (worker1):", remote_output.get())

# Ambil kembali model ke host
model.get()
print("Federated Learning simulation done.")
