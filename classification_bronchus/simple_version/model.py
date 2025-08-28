"""
简化版模型定义
"""

import torch
import torch.nn as nn
import timm
from torchvision import models
from config import Config

class BronchusClassifier(nn.Module):
    """支气管分类器"""
    
    def __init__(self, model_name=None, num_classes=None, pretrained=True):
        super(BronchusClassifier, self).__init__()
        
        self.model_name = model_name or Config.MODEL_NAME
        self.num_classes = num_classes or Config.NUM_CLASSES
        self.pretrained = pretrained
        
        self.backbone = self._create_backbone()
        self.classifier = self._create_classifier()
        
    def _create_backbone(self):
        """创建骨干网络"""
        if self.model_name.startswith('efficientnet'):
            # 使用timm库的EfficientNet
            model = timm.create_model(
                self.model_name,  # timm中使用下划线命名
                pretrained=self.pretrained,
                num_classes=0  # 不包含分类头
            )
            return model
            
        elif self.model_name.startswith('resnet'):
            # 使用torchvision的ResNet
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=self.pretrained)
            elif self.model_name == 'resnet101':
                model = models.resnet101(pretrained=self.pretrained)
            elif self.model_name == 'resnet152':
                model = models.resnet152(pretrained=self.pretrained)
            else:
                raise ValueError(f"不支持的ResNet模型: {self.model_name}")
            
            # 移除最后的分类层
            model = nn.Sequential(*list(model.children())[:-1])
            return model
            
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")
    
    def _create_classifier(self):
        """创建分类器"""
        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            features = self.backbone(dummy_input)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            feature_dim = features.shape[1]
        
        # 创建分类器
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
        
        return classifier
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        
        # 展平特征
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        output = self.classifier(features)
        return output

def create_model(model_name=None, num_classes=None, pretrained=True):
    """创建模型"""
    model = BronchusClassifier(model_name, num_classes, pretrained)
    model = model.to(Config.DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型: {model.model_name}")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    return model

if __name__ == "__main__":
    # 测试模型创建
    Config.print_config()
    model = create_model()
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(Config.DEVICE)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")