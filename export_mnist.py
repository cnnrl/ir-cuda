"""
Train a tiny MNIST MLP and export everything a C++/CUDA runtime needs:

  export/weights/<name>.bin   raw little-endian float32, row-major
  export/manifest.json        tensor names, shapes, files
  export/sample_input.bin     one normalized image, shape [1, 784], float32
  export/expected_logits.bin  PyTorch's output for that image, shape [1, 10]
  export/model.onnx           (optional) same model as ONNX

Model: 784 -> 128 -> ReLU -> 10
Linear weights are stored as [out_features, in_features], row-major,
so y = x @ W^T + b.

Usage:  pip install torch torchvision
        python export_mnist.py
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

OUT = "export" 
EPOCHS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):                 # x: [N, 784]
        return self.fc2(F.relu(self.fc1(x)))

# ---------------------------------------------------------------- data
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # standard MNIST stats
    transforms.Lambda(lambda t: t.view(-1)),      # flatten to 784
])
train = datasets.MNIST("data", train=True, download=True, transform=tf)
test = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_dl = torch.utils.data.DataLoader(test, batch_size=1000)

# ---------------------------------------------------------------- train
model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    print(f"epoch {epoch + 1}: test acc {correct / len(test):.4f}")

# ---------------------------------------------------------------- export
model.eval().cpu()
os.makedirs(f"{OUT}/weights", exist_ok=True)

manifest = {"tensors": []}
for name, p in model.state_dict().items():
    arr = p.detach().contiguous().numpy().astype("<f4")
    fname = f"weights/{name}.bin"
    arr.tofile(f"{OUT}/{fname}")
    manifest["tensors"].append(
        {"name": name, "shape": list(arr.shape), "dtype": "float32", "file": fname}
    )

# Sample input = first test image (already normalized + flattened)
x, label = test[0]
x = x.unsqueeze(0)                        # [1, 784]
x.numpy().astype("<f4").tofile(f"{OUT}/sample_input.bin")

with torch.no_grad():
    logits = model(x)
logits.numpy().astype("<f4").tofile(f"{OUT}/expected_logits.bin")

manifest["sample_input"] = {"file": "sample_input.bin", "shape": [1, 784]}
manifest["expected_logits"] = {"file": "expected_logits.bin", "shape": [1, 10]}
manifest["sample_label"] = int(label)
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Optional: ONNX export, handy if your IR wants to parse a graph
try:
    torch.onnx.export(model, x, f"{OUT}/model.onnx",
                      input_names=["input"], output_names=["logits"])
except Exception as e:
    print("ONNX export skipped:", e)

print(f"\nLabel: {label}  PyTorch prediction: {logits.argmax(1).item()}")
print("Expected logits:", np.round(logits.numpy()[0], 4))
print(f"Wrote everything to ./{OUT}/")
