# src/test.py
import torch
from torch.utils.data import DataLoader
from models.fusion_model import FusionModel
from src.dataset import DeepfakeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel().to(device)
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()

testset = DeepfakeDataset("data/")
dataloader = DataLoader(testset, batch_size=8, shuffle=False)

correct, total = 0, 0
with torch.no_grad():
    for img, dct, dlib, label in dataloader:
        img, dct, dlib, label = img.to(device), dct.to(device), dlib.to(device), label.to(device)
        output = model(img, dct, dlib)
        _, pred = output.max(1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Accuracy: {correct / total:.4f}")
