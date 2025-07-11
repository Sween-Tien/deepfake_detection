import numpy as np
import cv2
from scipy.fftpack import dct

# ==================== 配置参数 ====================
class Config:
    data_root = "data"
    batch_size = 32
    lr = 1e-4
    epochs = 50
    dct_block_size = 8
    num_dct_coeff = 15
    img_size = 224
    num_workers = 0
    weight_decay = 1e-4
    dropout_rate = 0.3
    patience = 5
    use_amp = True

# ==================== DCT特征提取函数 ====================
def extract_dct_feature(gray_img):
    features = []
    for i in range(0, Config.img_size, Config.dct_block_size):
        for j in range(0, Config.img_size, Config.dct_block_size):
            block = gray_img[i:i+Config.dct_block_size, j:j+Config.dct_block_size]
            if block.shape == (Config.dct_block_size, Config.dct_block_size):
                block_norm = (block - np.mean(block)) / (np.std(block) + 1e-6)
                dct_block = dct(dct(block_norm.T, norm='ortho').T, norm='ortho')
                features.extend(dct_block.flatten()[:Config.num_dct_coeff])
    return np.array(features) if features else np.zeros(Config.dct_block_size**2)
