import os
import cv2
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from skimage.feature import local_binary_pattern
import torch.cuda.amp as amp
import logging
from typing import Tuple, List, Optional

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    def __init__(self):
        self.feature_dim = 20  # 综合特征维度
        self._setup_feature_components()
        
    def _setup_feature_components(self):
        """预定义特征计算参数"""
        self.lbp_params = {
            'P': 8,       # 圆形邻域像素点数
            'R': 1,       # 邻域半径
            'method': 'uniform',
            'bins': 10,
            'range': (0, 10)
        }
        
    def _safe_image_read(self, img_path: str) -> Optional[np.ndarray]:
        """安全的图像读取方法"""
        try:
            with open(img_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"无法解码图像: {img_path}")
                    return None
                return img
        except Exception as e:
            logger.error(f"读取图像失败 {img_path}: {str(e)}")
            return None
    
    def _get_lbp_features(self, gray_img: np.ndarray) -> np.ndarray:
        """优化的LBP特征提取"""
        try:
            gray_uint8 = np.clip(gray_img, 0, 255).astype(np.uint8)
            lbp = local_binary_pattern(
                gray_uint8,
                self.lbp_params['P'],
                self.lbp_params['R'],
                method=self.lbp_params['method']
            )
            hist, _ = np.histogram(
                lbp,
                bins=self.lbp_params['bins'],
                range=self.lbp_params['range']
            )
            return hist / (hist.sum() + 1e-6)
        except Exception as e:
            logger.warning(f"LBP特征提取失败: {str(e)}")
            return np.zeros(self.lbp_params['bins'])
    
    def extract(self, img_path: str) -> np.ndarray:
        """主特征提取方法"""
        try:
            img = self._safe_image_read(img_path)
            if img is None:
                return np.zeros(self.feature_dim)
                
            return self._extract_features_from_image(img)
        except Exception as e:
            logger.error(f"特征提取异常 {img_path}: {str(e)}")
            return np.zeros(self.feature_dim)
    
    def extract_from_buffer(self, buffer: bytes) -> np.ndarray:
        """从内存缓冲区提取特征"""
        try:
            img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros(self.feature_dim)
                
            return self._extract_features_from_image(img)
        except Exception as e:
            logger.error(f"缓冲区特征提取失败: {str(e)}")
            return np.zeros(self.feature_dim)
    
    def _extract_features_from_image(self, img: np.ndarray) -> np.ndarray:
        """从已加载的图像中提取特征"""
        # 基础预处理
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. 频域特征
        dct = cv2.dct(gray - np.mean(gray))
        dct_feat = [np.mean(dct), np.std(dct), np.max(dct)]
        
        # 2. 梯度特征
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_feat = [np.mean(grad_mag), np.std(grad_mag)]
        
        # 3. 颜色特征
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_feat = [
            np.mean(hsv[:,:,0]),  # 色调
            np.std(hsv[:,:,1]),   # 饱和度
            np.median(hsv[:,:,2])  # 明度
        ]
        
        # 4. 纹理特征
        lbp_feat = self._get_lbp_features(gray)
        
        # 5. 图像质量特征
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        quality_feat = [blur_score, np.mean(img)]
        
        return np.concatenate([
            dct_feat,
            grad_feat,
            color_feat,
            lbp_feat,
            quality_feat
        ])

class RobustDataset(Dataset):
    def __init__(self, base_dir: str, phase: str = 'train'):
        self.extractor = AdvancedFeatureExtractor()
        self.phase = phase
        self.samples = []
        self.mean = None
        self.std = None
        
        self._load_and_validate_samples(base_dir)
        
        if phase == 'train' and len(self.samples) > 0:
            self._calculate_normalization_params()
    
    def _load_and_validate_samples(self, base_dir: str):
        """加载并验证样本"""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        sample_counts = {'real': 0, 'fake': 0}
        
        for label, folder in enumerate(['real', 'fake']):
            dir_path = os.path.join(base_dir, self.phase, folder)
            if not os.path.exists(dir_path):
                logger.warning(f"目录不存在: {dir_path}")
                continue
                
            for fname in os.listdir(dir_path):
                if fname.lower().endswith(valid_extensions):
                    img_path = os.path.join(dir_path, fname)
                    if self._validate_image(img_path):
                        self.samples.append((img_path, label))
                        sample_counts['real' if label == 0 else 'fake'] += 1
        
        logger.info(f"加载完成 - {self.phase}: 真实样本={sample_counts['real']}, 伪造样本={sample_counts['fake']}")
        if len(self.samples) == 0:
            raise ValueError(f"{self.phase} 数据集无有效样本！")
    
    def _validate_image(self, img_path: str) -> bool:
        """多层级图像验证"""
        # 基础检查
        if not os.path.exists(img_path) or os.path.getsize(img_path) < 1024:
            return False
            
        # 文件头检查
        try:
            with open(img_path, 'rb') as f:
                header = f.read(4)
                if not (header.startswith(b'\xff\xd8') or  # JPEG
                       header.startswith(b'\x89PNG')):     # PNG
                    return False
        except:
            return False
            
        # 内容检查
        try:
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                return False
            if len(img.shape) not in (2, 3):
                return False
            if np.mean(img) < 10 or np.mean(img) > 245:
                return False
            return True
        except:
            return False
    
    def _calculate_normalization_params(self):
        """计算标准化参数"""
        all_features = []
        for img_path, _ in tqdm(self.samples, desc="计算标准化参数"):
            features = self.extractor.extract(img_path)
            all_features.append(features)
        
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0) + 1e-6
    
    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """增强版数据增强"""
        # 随机组合增强技术
        transforms = [
            lambda x: cv2.flip(x, 1),  # 水平翻转
            lambda x: cv2.GaussianBlur(x, (3,3), 0),
            lambda x: cv2.addWeighted(x, 1.2, np.zeros_like(x), 0, -10)  # 对比度增强
        ]
        if np.random.rand() > 0.5:
            img = random.choice(transforms)(img)
        return img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        try:
            # 安全读取与增强
            with open(img_path, 'rb') as f:
                img_data = f.read()
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("图像解码失败")
            
            if self.phase == 'train':
                img = self._apply_augmentation(img)

            # 特征提取
            _, buf = cv2.imencode('.jpg', img)
            features = self.extractor.extract_from_buffer(buf.tobytes())

            if features is None or not isinstance(features, np.ndarray):
                raise ValueError("特征提取失败，返回 None")

            # 标准化
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / self.std

            return torch.FloatTensor(features), torch.tensor(label, dtype=torch.float)

        except Exception as e:
            logger.error(f"[跳过样本] {img_path}: {str(e)}")
            # 返回有效类型但无效数据，防止 DataLoader 崩溃
            return torch.zeros(self.extractor.feature_dim), torch.tensor(label, dtype=torch.float)

class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.Tanh(),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x)
        return x * weights

class EnhancedDetector(nn.Module):
    def __init__(self, input_dim: int = 20):
        super().__init__()
        self._build_network(input_dim)
        
    def _build_network(self, input_dim: int):
        """构建网络结构"""
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            AttentionBlock(32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return self.classifier(features).squeeze()

class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    
    def should_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    model_save_path: str
) -> Tuple[float, float]:
    """训练模型并返回最佳验证指标"""
    scaler = amp.GradScaler()
    best_auc = 0.0
    best_acc = 0.0
    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # 训练阶段
        for features, labels in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        logger.info(f"Epoch {epoch+1}: 训练损失 = {avg_train_loss:.4f}")
        
        # 验证阶段
        val_acc, val_auc = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: 验证准确率 = {val_acc:.4f}, AUC = {val_auc:.4f}")
        
        # 检查是否是最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"新最佳模型已保存 (AUC={best_auc:.4f})")
        
        # 早停检查
        if early_stopper.should_stop(1 - val_auc):  # 使用1-AUC作为损失
            logger.info("早停触发，停止训练")
            break
    
    return best_acc, best_auc

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """评估模型并返回准确率和AUC"""
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features)
            preds.extend(outputs.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    preds = np.clip(preds, 1e-6, 1-1e-6)  # 避免log(0)
    acc = accuracy_score(np.round(preds), trues)
    auc = roc_auc_score(trues, preds)
    
    return acc, auc

def check_data_directory(data_dir: str) -> bool:
    """检查数据目录结构是否正确"""
    required = ['train/real', 'train/fake', 'val/real', 'val/fake']
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing:
        logger.error("缺少以下目录:\n" + "\n".join(missing))
        logger.info("需要的目录结构:\ndata/\n├── train/\n│   ├── real/\n│   └── fake/\n"
                  "├── val/\n│   ├── real/\n│   └── fake/")
        return False
    return True

def main():
    # 配置参数
    config = {
        'data_dir': "data",
        'num_epochs': 30,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'model_save_path': "best_model.pth"
    }
    
    # 检查数据目录
    if not check_data_directory(config['data_dir']):
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    try:
        # 加载数据
        train_set = RobustDataset(config['data_dir'], 'train')
        val_set = RobustDataset(config['data_dir'], 'val')
        
        train_loader = DataLoader(
            train_set,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # 初始化模型
        model = EnhancedDetector(input_dim=train_set.extractor.feature_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.BCELoss()
        
        # 训练模型
        best_acc, best_auc = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config['num_epochs'],
            model_save_path=config['model_save_path']
        )
        
        logger.info(f"训练完成。最佳验证准确率: {best_acc:.4f}, AUC: {best_auc:.4f}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()