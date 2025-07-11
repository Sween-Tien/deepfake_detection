import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据集类
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = list(Path(real_dir).glob("*"))
        self.fake_images = list(Path(fake_dir).glob("*"))
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# 改进的ResNet模型
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结所有卷积层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换分类头
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )
    return model.to(device)

# 训练函数
def train(model, train_loader, val_loader, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        val_acc = evaluate(model, val_loader)
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
    
    return model

# 评估函数
def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs).squeeze())
            predicted = (outputs > 0.5).int()
            correct += (predicted == labels).sum().item()
    return correct / len(loader.dataset)

if __name__ == "__main__":
    # 数据路径
    data_dir = Path("data")
    train_real = data_dir / "train" / "real"
    train_fake = data_dir / "train" / "fake"
    val_real = data_dir / "val" / "real"
    val_fake = data_dir / "val" / "fake"
    
    # 加载数据
    train_dataset = DeepfakeDataset(train_real, train_fake, train_transform)
    val_dataset = DeepfakeDataset(val_real, val_fake, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"训练集: {len(train_dataset)}张 (正负样本比 {len(train_dataset.real_images)}:{len(train_dataset.fake_images)})")
    print(f"验证集: {len(val_dataset)}张 (正负样本比 {len(val_dataset.real_images)}:{len(val_dataset.fake_images)})")
    
    # 初始化模型
    model = get_model()
    print(model)
    
    # 训练
    model = train(model, train_loader, val_loader)
    
    # 测试最佳模型
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc = evaluate(model, val_loader)
    print(f"\n最终验证准确率: {test_acc:.4f}")