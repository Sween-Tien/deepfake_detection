import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 去掉FC层

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class DeepfakeDetector(nn.Module):
    def __init__(self, dct_dim=64, dlib_dim=136):
        super().__init__()
        self.resnet = ResNetFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + dct_dim + dlib_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, image, dct_feat, dlib_feat):
        resnet_feat = self.resnet(image)
        fused = torch.cat([resnet_feat, dct_feat, dlib_feat], dim=1)
        return self.classifier(fused)
