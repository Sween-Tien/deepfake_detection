import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import (accuracy_score, roc_auc_score, recall_score, 
                            f1_score, precision_score, confusion_matrix, 
                            roc_curve, precision_recall_curve)
import seaborn as sns
import os
import shutil
import hashlib
from collections import defaultdict

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# 数据准备模块
# --------------------------
class DataPreprocessor:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def split_dataset(self, original_dir, output_dir="data", ratios=(0.7, 0.15, 0.15), seed=42):
        """划分数据集为train/val/test"""
        assert sum(ratios) == 1.0, "比例总和必须为1"
        random.seed(seed)
        
        output_dir = Path(output_dir)
        splits = ["train", "val", "test"]
        for split in splits:
            for cls in ["real", "fake"]:
                (output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        for class_name in ["real", "fake"]:
            class_path = Path(original_dir) / class_name
            if not class_path.exists():
                raise FileNotFoundError(f"目录不存在: {class_path}")
                
            files = [f for f in class_path.glob("*") if f.is_file()]
            random.shuffle(files)
            
            n_total = len(files)
            n_train = int(n_total * ratios[0])
            n_val = int(n_total * ratios[1])
            
            train_files = files[:n_train]
            val_files = files[n_train:n_train+n_val]
            test_files = files[n_train+n_val:]
            
            print(f"\n处理类别: {class_name} (共{n_total}张图片)")
            for split, split_files in zip(splits, [train_files, val_files, test_files]):
                print(f"  {split}: {len(split_files)}张")
                for src_file in tqdm(split_files, desc=f"复制 {split} 数据"):
                    dst_dir = output_dir / split / class_name
                    shutil.copy2(src_file, dst_dir / src_file.name)

# --------------------------
# 数据集类
# --------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_images = self._load_images(real_dir)
        self.fake_images = self._load_images(fake_dir)
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.transform = transform
        
    def _load_images(self, dir_path):
        return [f for f in Path(dir_path).glob("*") 
               if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# --------------------------
# 模型模块
# --------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载ResNet50基础模型
        try:
            # 尝试自动下载权重
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
        except RuntimeError as e:
            print(f"自动下载权重失败: {str(e)}")
            print("尝试手动下载...")
            base_model = self._load_manually(pretrained)
                
        # 冻结前3个stage的参数
        for name, param in base_model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
                
        # 替换分类头
        base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )
        self.model = base_model
        
    def _load_manually(self, pretrained):
        """手动加载预训练权重"""
        base_model = models.resnet50(weights=None)
        if not pretrained:
            return base_model
            
        WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        SAVE_PATH = "../models/pretrained/resnet50_weights.pth"
        
        # 下载权重文件
        if not os.path.exists(SAVE_PATH):
            print(f"正在下载权重文件到: {SAVE_PATH}")
            torch.hub.download_url_to_file(WEIGHTS_URL, SAVE_PATH)
        
        # 验证文件完整性
        expected_hash = "0676ba61"
        with open(SAVE_PATH, "rb") as f:
            actual_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
        if actual_hash != expected_hash:
            raise ValueError(f"权重文件损坏！期望MD5前8位: {expected_hash}, 实际: {actual_hash}")
        
        # 加载权重
        base_model.load_state_dict(torch.load(SAVE_PATH))
        return base_model
        
    def forward(self, x):
        return self.model(x)

# --------------------------
# 训练和评估模块
# --------------------------
class Trainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.best_metrics = {
            'Accuracy': 0,
            'AUC-ROC': 0,
            'Recall': 0,
            'F1-Score': 0
        }
        self.history = defaultdict(list)
        
    def train(self, train_loader, val_loader, epochs=15):
        # 计算类别权重
        pos_weight = torch.tensor([
            len(train_loader.dataset.real_images) / len(train_loader.dataset.fake_images)
        ]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = torch.optim.AdamW([
            {'params': self.model.model.layer4.parameters(), 'lr': 1e-4},
            {'params': self.model.model.fc.parameters(), 'lr': 3e-4}
        ], weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for inputs, labels in progress:
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                progress.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # 验证并保存历史记录
            val_metrics = self.evaluate(val_loader)
            for k, v in val_metrics.items():
                if k != 'Confusion_Matrix':
                    self.history[k].append(v)
            
            # 更新最佳指标
            if val_metrics['F1-Score'] > self.best_metrics['F1-Score']:
                self.best_metrics = val_metrics.copy()
                torch.save(self.model.state_dict(), "best_model.pth")
                print(f"Saved best model with F1-Score: {val_metrics['F1-Score']:.4f}")
            
            # 学习率调整
            scheduler.step(val_metrics['F1-Score'])
            
            # 打印epoch结果
            self._print_epoch_results(epoch, train_loss/len(train_loader), val_metrics)
        
        # 训练完成后绘制历史曲线
        self._plot_training_history()
    
    def evaluate(self, loader):
        self.model.eval()
        y_true, y_pred, y_score = [], [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                
                # 收集预测结果
                y_true.extend(labels.cpu().numpy())
                y_pred.extend((torch.sigmoid(outputs) > 0.5).int().cpu().numpy())
                y_score.extend(torch.sigmoid(outputs).cpu().numpy())
        
        # 计算所有指标
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUC-ROC': roc_auc_score(y_true, y_score),
            'Recall': recall_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'Confusion_Matrix': confusion_matrix(y_true, y_pred)
        }
        
        # 可视化结果
        self._plot_metrics(metrics, y_true, y_score)
        
        return metrics
    
    def _print_epoch_results(self, epoch, train_loss, val_metrics):
        """打印epoch训练结果"""
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Metrics:")
        print(f"    Accuracy: {val_metrics['Accuracy']:.4f}")
        print(f"    AUC-ROC: {val_metrics['AUC-ROC']:.4f}")
        print(f"    Precision: {val_metrics['Precision']:.4f}")
        print(f"    Recall: {val_metrics['Recall']:.4f}")
        print(f"    F1-Score: {val_metrics['F1-Score']:.4f}")
        print(f"    Confusion Matrix:\n{val_metrics['Confusion_Matrix']}")
    
    def _plot_metrics(self, metrics, y_true, y_score):
        """绘制评估指标可视化"""
        plt.figure(figsize=(18, 5))
        
        # 指标条形图
        plt.subplot(1, 4, 1)
        bars = plt.bar(['Accuracy', 'Precision', 'Recall', 'F1'], 
                      [metrics['Accuracy'], metrics['Precision'], 
                       metrics['Recall'], metrics['F1-Score']],
                      color=['blue', 'green', 'orange', 'red'])
        plt.ylim(0, 1)
        plt.title('Classification Metrics')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # ROC曲线
        plt.subplot(1, 4, 2)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC-ROC"]:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # PR曲线
        plt.subplot(1, 4, 3)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.plot(recall, precision, label=f'F1 = {metrics["F1-Score"]:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # 混淆矩阵
        plt.subplot(1, 4, 4)
        sns.heatmap(metrics['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Pred Real', 'Pred Fake'],
                   yticklabels=['True Real', 'True Fake'])
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f'epoch_metrics.png')
        plt.close()
    
    def _plot_training_history(self):
        """绘制训练历史曲线"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['Accuracy', 'Recall', 'F1-Score', 'AUC-ROC']
        colors = ['b', 'g', 'r', 'm']
        
        for idx, metric in enumerate(metrics):
            plt.plot(self.history[metric], 
                    label=f'{metric} (max={max(self.history[metric]):.4f})',
                    color=colors[idx], marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()
    
    def test(self, test_loader):
        """在测试集上完整评估"""
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(test_loader)
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {test_metrics['Accuracy']:.4f}")
        print(f"AUC-ROC: {test_metrics['AUC-ROC']:.4f}")
        print(f"Precision: {test_metrics['Precision']:.4f}")
        print(f"Recall: {test_metrics['Recall']:.4f}")
        print(f"F1-Score: {test_metrics['F1-Score']:.4f}")
        
        # 保存详细报告
        with open("test_report.txt", "w") as f:
            f.write("Deepfake Detection Model Test Report\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy: {test_metrics['Accuracy']:.4f}\n")
            f.write(f"AUC-ROC: {test_metrics['AUC-ROC']:.4f}\n")
            f.write(f"Precision: {test_metrics['Precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['Recall']:.4f}\n") 
            f.write(f"F1-Score: {test_metrics['F1-Score']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(test_metrics['Confusion_Matrix']))
        
        # 绘制最佳阈值分析
        self._plot_threshold_analysis(test_loader)
    
    def _plot_threshold_analysis(self, loader):
        """绘制阈值分析图"""
        self.model.eval()
        y_true, y_score = [], []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs).squeeze()
                y_true.extend(labels.cpu().numpy())
                y_score.extend(torch.sigmoid(outputs).cpu().numpy())
        
        # 计算不同阈值下的指标
        thresholds = np.linspace(0.1, 0.9, 50)
        accuracies = []
        recalls = []
        precisions = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (np.array(y_score) > thresh).astype(int)
            accuracies.append(accuracy_score(y_true, y_pred))
            recalls.append(recall_score(y_true, y_pred))
            precisions.append(precision_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred))
        
        # 绘制阈值分析
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, label='Accuracy')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=3)
        
        # 标记最佳F1阈值
        best_idx = np.argmax(f1_scores)
        plt.scatter(thresholds[best_idx], f1_scores[best_idx], 
                   c='red', s=100, label=f'Best F1 (t={thresholds[best_idx]:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('threshold_analysis.png')
        plt.close()

# --------------------------
# 主程序
# --------------------------
def main():
    # 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 如果需要重新划分数据
    if not Path("data/train").exists():
        print("正在划分数据集...")
        preprocessor.split_dataset(
            original_dir="original_data",
            output_dir="data",
            ratios=(0.7, 0.15, 0.15),
            seed=42
        )
    
    # 数据路径
    data_dir = Path("data")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    # 创建数据集
    train_dataset = DeepfakeDataset(
        train_dir / "real", 
        train_dir / "fake",
        preprocessor.train_transform
    )
    val_dataset = DeepfakeDataset(
        val_dir / "real",
        val_dir / "fake",
        preprocessor.val_transform
    )
    test_dataset = DeepfakeDataset(
        test_dir / "real",
        test_dir / "fake",
        preprocessor.val_transform
    )
    
    print(f"\n训练集样本: {len(train_dataset)} (正负样本比 {len(train_dataset.real_images)}:{len(train_dataset.fake_images)})")
    print(f"验证集样本: {len(val_dataset)} (正负样本比 {len(val_dataset.real_images)}:{len(val_dataset.fake_images)})")
    print(f"测试集样本: {len(test_dataset)} (正负样本比 {len(test_dataset.real_images)}:{len(test_dataset.fake_images)})")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = DeepfakeDetector(pretrained=True)
    print(f"\n可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, epochs=15)
    
    # 测试
    trainer.test(test_loader)
    
    print("\n训练完成！所有评估结果已保存到当前目录")

if __name__ == "__main__":
    main()