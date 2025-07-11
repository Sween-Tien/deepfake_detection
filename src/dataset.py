# src/dataset.py
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import torch
from features.dct_utils import extract_dct_feature
from features.dlib_utils import extract_geometric_features

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 默认float32
        ])
        for label, sub in enumerate(["real", "fake"]):
            subdir = os.path.join(root_dir, sub)
            for fname in os.listdir(subdir):
                if fname.endswith(".jpg") or fname.endswith(".png"):
                    self.data.append((os.path.join(subdir, fname), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        # 读取PIL图像，转Tensor
        image = Image.open(path).convert("RGB")
        img_tensor = self.transform(image)  # float32 tensor

        # 读取cv2图像用于DCT特征提取（保持RGB）
        cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        # 提取DCT特征，转为float32 tensor
        dct_feat_np = extract_dct_feature(cv_img)
        dct_feat = torch.tensor(dct_feat_np, dtype=torch.float32)

        # 提取Dlib特征，传入路径，转为float32 tensor
        dlib_feat_np = extract_geometric_features(path)
        dlib_feat = torch.tensor(dlib_feat_np, dtype=torch.float32)

        return img_tensor, dct_feat, dlib_feat, label
