# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.fusion_model import FusionModel
from src.dataset import DeepfakeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel().to(device)

dataset = DeepfakeDataset("data/")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total, correct = 0, 0
    for img, dct, dlib, label in dataloader:
        img, dct, dlib, label = img.to(device), dct.to(device), dlib.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(img, dct, dlib)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        _, pred = output.max(1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    print(f"Epoch {epoch+1}: Acc={correct/total:.4f}")

torch.save(model.state_dict(), "checkpoints/model.pth")
