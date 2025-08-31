# model.py
import torch
from torch import nn
import common
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn

class mymodel(nn.Module):
    def __init__(self, num_chars=52, captcha_size=4):
        super(mymodel, self).__init__()
        self.captcha_size = captcha_size
        
        # 增强的卷积特征提取器
        self.conv_layers = nn.Sequential(
            # 输入: [batch, 1, 60, 160]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 30, 80]
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 15, 40]
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 64, 7, 20]
            
            # 使用空洞卷积扩大感受野
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 128, 3, 10]
        )
        
        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [batch, 128, 1, 1]
        
        # 注意力机制增强特征提取
        self.attention = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.Sigmoid()
        )
        
        # 特征嵌入层 (用于域适应)
        self.feature_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # 分类层 (用于主任务)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
        )
        
        # 每个字符一个输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(512, num_chars) for _ in range(captcha_size)
        ])
    
    def extract_features(self, x):
        """提取特征用于域适应"""
        # 卷积特征提取
        conv_features = self.conv_layers(x)
        
        # 应用注意力机制
        attention_mask = self.attention(conv_features)
        attended_features = conv_features * attention_mask
        
        # 全局池化
        pooled_features = self.global_pool(attended_features)
        
        # 特征嵌入
        embedded_features = self.feature_embedding(pooled_features)
        return embedded_features
    
    def classify(self, features):
        """基于特征进行分类"""
        # 分类处理
        classifier_out = self.classifier(features)
        
        # 为每个字符位置生成预测
        outputs = []
        for i in range(self.captcha_size):
            outputs.append(self.output_layers[i](classifier_out))
        
        # 堆叠结果: [batch, captcha_size, num_classes]
        return torch.stack(outputs, dim=1)
    
    def forward(self, x):
        # 特征提取
        features = self.extract_features(x)
        
        # 分类
        return self.classify(features)
if __name__ == '__main__':
    data=torch.ones(64,1,60,160)
    model=mymodel()
    x=model(data)
    print(x.shape)