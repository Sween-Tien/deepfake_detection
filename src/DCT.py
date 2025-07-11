import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report)
from tqdm import tqdm
from scipy.fftpack import dct
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ==================== 配置参数 ====================
class Config:
    # 数据与路径
    data_root = "data"
    img_size = 256
    batch_size = 16
    
    # 训练超参数
    lr = 3e-4
    epochs = 100
    weight_decay = 1e-4
    patience = 10
    
    # DCT参数
    dct_block_size = 8
    num_dct_coeff = 20
    
    # 模型结构
    dropout_rate = 0.5
    use_amp = True
    
    # 系统配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if os.name == 'nt' else 4  # Windows兼容

# ==================== 数据加载 ====================
class DeepfakeDataset(Dataset):
    def __init__(self, phase="train"):
        self.phase = phase
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(Config.data_root, self.phase, cls)
            if not os.path.exists(cls_dir):
                continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    samples.append((img_path, label))
        
        real_count = sum(1 for _, label in samples if label == 0)
        fake_count = len(samples) - real_count
        print(f"{self.phase} set - Real: {real_count}, Fake: {fake_count}")
        return samples
    
    def _augment_image(self, img):
        """Windows兼容的数据增强"""
        # 颜色扰动
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:,:,0] = (img[:,:,0] + random.randint(-15,15)) % 180
        img[:,:,1] = np.clip(img[:,:,1] * random.uniform(0.7,1.3), 0, 255)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        
        # 几何变换
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
                
            img = cv2.resize(img, (Config.img_size, Config.img_size))
            
            if self.phase == "train":
                img = self._augment_image(img)
            
            # DCT特征提取
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = []
            for i in range(0, Config.img_size, Config.dct_block_size):
                for j in range(0, Config.img_size, Config.dct_block_size):
                    block = gray[i:i+Config.dct_block_size, j:j+Config.dct_block_size]
                    if block.shape == (Config.dct_block_size, Config.dct_block_size):
                        block_norm = (block - np.mean(block)) / (np.std(block) + 1e-6)
                        dct_block = dct(dct(block_norm.T, norm='ortho').T, norm='ortho')
                        features.extend(dct_block.flatten()[:Config.num_dct_coeff])
            
            dct_feat = np.array(features) if features else np.zeros(Config.dct_block_size**2)
            
            img_tensor = torch.FloatTensor(img.transpose(2,0,1)/255.0)
            dct_tensor = torch.FloatTensor(dct_feat)
            
            return img_tensor, dct_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")
            zero_img = torch.zeros(3, Config.img_size, Config.img_size)
            zero_feat = torch.zeros(Config.dct_block_size**2)
            return zero_img, zero_feat, torch.tensor(label, dtype=torch.float32)

# ==================== 模型定义 ====================
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class DCTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN图像特征提取
        self.cnn = nn.Sequential(
            BasicBlock(3, 64),
            nn.MaxPool2d(2),
            BasicBlock(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # DCT特征处理
        dct_feat_dim = (Config.img_size//Config.dct_block_size)**2 * Config.num_dct_coeff
        self.dct_fc = nn.Sequential(
            nn.Linear(dct_feat_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 + 512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, dct_feat):
        cnn_feat = self.cnn(img).view(img.size(0), -1)
        dct_feat = self.dct_fc(dct_feat)
        combined = torch.cat([cnn_feat, dct_feat], dim=1)
        return self.classifier(combined)

# ==================== 训练函数 ====================
def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    for img, dct_feat, label in tqdm(dataloader, desc="Training"):
        img = img.to(device)
        dct_feat = dct_feat.to(device)
        label = label.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', enabled=Config.use_amp):
            outputs = model(img, dct_feat)
            loss = criterion(outputs, label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# ==================== 评估函数 ====================
def full_evaluation(y_true, y_pred, y_score=None, model_name=""):
    """完整评估模型性能"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_score is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_score)
    
    # 打印分类报告
    print(f"\n{model_name} 详细评估:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(6,5))
    sns.heatmap(metrics['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Pred Real', 'Pred Fake'],
               yticklabels=['True Real', 'True Fake'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    return metrics

def evaluate(model, dataloader, device, phase="Validation"):
    model.eval()
    preds, labels, scores = [], [], []
    
    with torch.no_grad():
        for img, dct_feat, label in tqdm(dataloader, desc=f"{phase} Evaluating"):
            img = img.to(device)
            dct_feat = dct_feat.to(device)
            outputs = model(img, dct_feat).cpu().squeeze()
            
            scores.extend(outputs.numpy())
            preds.extend((outputs > 0.5).int().numpy())
            labels.extend(label.cpu().numpy())
    
    return full_evaluation(labels, preds, scores, model_name=f"{phase} Set")

# ==================== 主程序 ====================
def main():
    # 检查数据目录
    required_dirs = ['train/real', 'train/fake', 'val/real', 'val/fake']
    missing = [d for d in required_dirs if not os.path.exists(os.path.join(Config.data_root, d))]
    if missing:
        print(f"缺少目录: {missing}")
        return
    
    # 初始化模型
    model = DCTDetector().to(Config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(Config.device))
    optimizer = optim.AdamW(model.parameters(), 
                          lr=Config.lr,
                          weight_decay=Config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)
    
    # 数据加载
    train_set = DeepfakeDataset("train")
    val_set = DeepfakeDataset("val")
    test_set = DeepfakeDataset("test")
    
    # 处理类别不平衡
    weights = [2 if label == 1 else 1 for _, label in train_set.samples]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        sampler=sampler,
        num_workers=Config.num_workers
    )
    val_loader = DataLoader(
        val_set,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )
    
    # 训练监控
    history = defaultdict(list)
    best_metrics = None
    no_improve = 0
    
    for epoch in range(Config.epochs):
        # 训练阶段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, Config.device)
        history['Train Loss'].append(train_loss)
        
        # 验证阶段
        val_metrics = evaluate(model, val_loader, Config.device, "Validation")
        for k, v in val_metrics.items():
            if k != 'Confusion_Matrix':
                history[k].append(v)
        
        # 早停与模型保存
        if best_metrics is None or val_metrics['F1-Score'] > best_metrics['F1-Score']:
            best_metrics = val_metrics
            torch.save(model.state_dict(), "best_model.pth")
            print("※ 保存最佳模型 ※")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print(f"早停触发，已{Config.patience}轮无改善")
                break
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['Accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['Precision']:.4f}")
        print(f"Val Recall: {val_metrics['Recall']:.4f}")
        print(f"Val F1-Score: {val_metrics['F1-Score']:.4f}")
        print(f"Val AUC-ROC: {val_metrics.get('AUC-ROC', 0):.4f}")
    
    # 最终测试
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = evaluate(model, test_loader, Config.device, "Test")
    
    print("\n=== 最终测试结果 ===")
    print(f"Accuracy: {test_metrics['Accuracy']:.4f}")
    print(f"Precision: {test_metrics['Precision']:.4f}")
    print(f"Recall: {test_metrics['Recall']:.4f}")
    print(f"F1-Score: {test_metrics['F1-Score']:.4f}")
    print(f"AUC-ROC: {test_metrics.get('AUC-ROC', 0):.4f}")
    
    # 可视化训练过程
    plt.figure(figsize=(15,5))
    
    # 损失曲线
    plt.subplot(1,2,1)
    plt.plot(history['Train Loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 指标曲线
    plt.subplot(1,2,2)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        if metric in history:
            plt.plot(history[metric], label=metric)
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    main()