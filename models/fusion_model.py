# models/fusion_model.py
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        # 使用预训练的ResNet50，去掉最后的FC层
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # 特征维度：resnet50(2048) + DCT(512) + dlib(136)
        self.fc1 = nn.Linear(2048 + 512 + 136, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, img, dct_feat, dlib_feat):
        x1 = self.resnet(img)                 # (B, 2048)
        x2 = dct_feat                         # (B, 512)
        x3 = dlib_feat                        # (B, 136)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fc1(x)
        x = self.dropout(torch.relu(x))
        return self.fc2(x)
