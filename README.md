# Ai-model
Berikut adalah **penjelasan lengkap dari semua kode** yang bisa langsung kamu salin ke file `README.md` untuk GitHub. Ini sudah disusun rapi berdasarkan struktur topik presentasi kamu:

---

# **AI Model Presentation Code**

Kode ini berisi implementasi sederhana dari 4 topik utama dalam pembelajaran mesin dan AI:

1. Linear Regression  
2. Two-Layer Neural Network  
3. Optimization  
4. AI in Privacy (Federated Learning)

---

## **1. Linear Regression**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
lin_reg = LinearRegression().fit(X, y)

print("Linear Regression Coef:", lin_reg.coef_)
```

**Penjelasan:**  
Membuat model regresi linier dari data `X` dan `y`, di mana hubungan antara input dan output adalah linear (`y = 2x`). Model akan mempelajari koefisien tersebut secara otomatis.

---

## **2. Two-Layer Neural Network**

```python
import torch
import torch.nn as nn

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
```

**Penjelasan:**  
Model neural network dengan 1 hidden layer dan aktivasi ReLU. Ini adalah jaringan saraf sederhana yang cocok untuk masalah regresi kecil.

---

## **3. Optimisation & Training**

```python
import torch.optim as optim

model = TwoLayerNN(1, 5, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
targets = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

print("Loss after training:", loss.item())
```

**Penjelasan:**  
Melatih model neural network menggunakan SGD (Stochastic Gradient Descent) dengan Mean Squared Error sebagai fungsi loss. Setelah 200 epoch, model bisa mendekati fungsi `y = 2x`.

---

## **4. AI in Privacy: Federated Learning Simulation**

```python
import syft as sy
from syft.frameworks.torch import TorchHook
from syft.workers.virtual import VirtualWorker

hook = TorchHook(torch)
worker1 = VirtualWorker(hook, id="worker1")
worker2 = VirtualWorker(hook, id="worker2")

data = torch.tensor([[1.0], [2.0]]).send(worker1)
label = torch.tensor([[2.0], [4.0]]).send(worker1)

model.send(worker1)
remote_output = model(data)
print("Remote output (worker1):", remote_output.get())

model.get()
print("Federated Learning simulation done.")
```

**Penjelasan:**  
Simulasi federated learning di mana data tidak dikirim ke server pusat, melainkan model yang dikirim ke node (worker). Ini menjaga privasi data dan digunakan dalam aplikasi nyata seperti aplikasi kesehatan atau keyboard smartphone.

---

## **Visualisasi**

Gambar-gambar berikut dapat digunakan dalam presentasi:

- **Linear Regression**: Scatter plot + garis prediksi  
- **Loss Curve**: Kurva penurunan loss saat training  
- (Opsional) **Diagram Federated Learning** dan **Struktur Neural Network**

---

## **Dependencies**

Pastikan Anda menginstal pustaka berikut:

```bash
pip install numpy torch scikit-learn matplotlib syft==0.2.9
```

---

Kalau kamu ingin, saya juga bisa bantu buatkan versi `README.md` langsung sebagai file. Mau?
