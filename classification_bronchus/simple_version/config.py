"""
简化版配置文件
"""

import os
import torch

class Config:
    """统一配置类"""
    
    # 数据配置
    DATA_ROOT = "../data/data_split"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # 模型配置
    MODEL_NAME = "efficientnet_b0"  # 支持: efficientnet_b0, resnet50, resnet101
    NUM_CLASSES = 33
    PRETRAINED = True
    
    # 训练配置
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 数据加载配置
    NUM_WORKERS = 4
    IMAGE_SIZE = 224
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    WEIGHT_DIR = "../weight"
    LOG_DIR = "../logs"
    
    # 数据增强配置
    USE_AUGMENTATION = True
    
    @classmethod
    def create_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.WEIGHT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("配置信息:")
        print(f"模型: {cls.MODEL_NAME}")
        print(f"类别数: {cls.NUM_CLASSES}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"学习率: {cls.LEARNING_RATE}")
        print(f"训练轮数: {cls.NUM_EPOCHS}")
        print(f"设备: {cls.DEVICE}")
        print(f"图像尺寸: {cls.IMAGE_SIZE}")
        print("=" * 50)