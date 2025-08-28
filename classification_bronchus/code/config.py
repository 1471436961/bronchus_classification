"""
项目配置管理
定义数据、模型、训练等各模块的配置参数
"""

import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import torch
from torchvision import models
import timm
from utils.constants import *
from utils.validation_utils import validate_config, ValidationError
from utils.logger_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """数据相关配置"""
    # 路径配置
    data_root: str = "../data"
    train_path: str = "../data/data_split/train"
    val_path: str = "../data/data_split/val"
    test_path: str = "../data/data_split/test"
    
    # 数据集划分配置
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_samples_per_class: int = 10
    
    # 图像配置
    input_size: Tuple[int, int] = (224, 224)
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 数据增强配置
    augmentation_level: str = "medium"  # light, medium, heavy
    use_custom_normalization: bool = False  # 是否使用数据集特定的标准化参数
    
    # 数据加载配置
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: Optional[int] = None  # None表示自动检测
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # 类别平衡配置
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    
    # 缓存配置
    cache_images: bool = False  # 是否缓存图像到内存
    cache_transforms: bool = True  # 是否缓存变换结果
    
    def __post_init__(self):
        """配置后处理和验证"""
        if self.num_workers is None:
            self.num_workers = min(DEFAULT_NUM_WORKERS, os.cpu_count() or 1)
        
        if not (MIN_BATCH_SIZE <= self.batch_size <= MAX_BATCH_SIZE):
            raise ValidationError(f"batch_size必须在{MIN_BATCH_SIZE}-{MAX_BATCH_SIZE}之间")
        
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValidationError(f"数据集划分比例总和必须为1.0，当前为{total_ratio}")
        
        os.makedirs(self.data_root, exist_ok=True)
        logger.info(f"数据配置验证通过: batch_size={self.batch_size}, num_workers={self.num_workers}")


@dataclass
class ModelConfig:
    """模型相关配置"""
    # 模型选择
    model_name: str = "efficientnet_b0"  # efficientnet_b0, convnext_tiny, resnet50, etc.
    num_classes: int = 33
    pretrained: bool = True
    
    # 模型特定配置
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1
    
    # 输入尺寸映射
    model_input_sizes: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'efficientnet_b0': (224, 224),
        'efficientnet_b3': (300, 300),
        'efficientnet_b7': (600, 600),
        'convnext_tiny': (224, 224),
        'convnext_small': (224, 224),
        'resnet50': (224, 224),
        'resnet101': (224, 224),
        'vit_base_patch16_224': (224, 224),
        'swin_tiny_patch4_window7_224': (224, 224),
    })
    
    def get_input_size(self) -> Tuple[int, int]:
        """获取模型对应的输入尺寸"""
        return self.model_input_sizes.get(self.model_name, (224, 224))
    
    def create_model(self):
        """创建模型实例"""
        if self.model_name.startswith('efficientnet'):
            if self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=self.pretrained)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, self.num_classes)
            elif self.model_name == 'efficientnet_b3':
                model = models.efficientnet_b3(pretrained=self.pretrained)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, self.num_classes)
            else:
                # 使用timm创建其他EfficientNet变体
                model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
                
        elif self.model_name.startswith('convnext'):
            if self.model_name == 'convnext_tiny':
                model = models.convnext_tiny(pretrained=self.pretrained)
                model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, self.num_classes)
            else:
                model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
                
        elif self.model_name.startswith('resnet'):
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=self.pretrained)
                model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            elif self.model_name == 'resnet101':
                model = models.resnet101(pretrained=self.pretrained)
                model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            else:
                model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
                
        else:
            # 使用timm创建其他模型
            model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
        
        return model


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础训练参数
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # 训练器配置
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9  # 仅用于SGD
    
    # 学习率调度
    scheduler: str = "cosine"  # cosine, step, plateau, exponential
    step_size: int = 10  # 用于StepLR
    gamma: float = 0.1  # 用于StepLR和ExponentialLR
    milestones: List[int] = field(default_factory=lambda: [15, 30, 45])  # 用于MultiStepLR
    patience: int = 5  # 用于ReduceLROnPlateau
    
    # 损失函数配置
    loss_function: str = "cross_entropy"  # cross_entropy, focal_loss, label_smoothing
    label_smoothing: float = 0.1
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    
    # 正则化配置
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    # 训练技巧
    use_amp: bool = True  # 混合精度训练
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    
    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 检查点配置
    save_top_k: int = 3
    save_last: bool = True
    monitor_metric: str = "val_acc"  # val_acc, val_loss
    monitor_mode: str = "max"  # max, min


@dataclass
class SystemConfig:
    """系统相关配置"""
    # 设备配置
    device: str = "auto"  # auto, cuda, cpu
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # 随机种子
    seed: int = 42
    deterministic: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_every_n_steps: int = 50
    
    # 输出路径
    output_dir: str = "../weight"
    log_dir: str = "../logs"
    
    def __post_init__(self):
        """后处理"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """完整的训练配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """后处理，同步相关配置"""
        # 同步输入尺寸
        self.data.input_size = self.model.get_input_size()
        
        # 同步类别数
        self.model.num_classes = self.data.num_classes if hasattr(self.data, 'num_classes') else 33
    
    def save_config(self, path: str):
        """保存配置到文件"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, path: str):
        """从文件加载配置"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            system=SystemConfig(**config_dict['system'])
        )


# 预定义配置模板
class ConfigTemplates:
    """预定义的配置模板"""
    
    @staticmethod
    def get_fast_training_config() -> Config:
        """快速训练配置（用于验证）"""
        config = Config()
        config.training.epochs = 10
        config.training.early_stopping_patience = 3
        config.data.batch_size = 32
        config.data.augmentation_level = "light"
        return config
    
    @staticmethod
    def get_high_quality_config() -> Config:
        """高质量训练配置"""
        config = Config()
        config.training.epochs = 100
        config.training.learning_rate = 5e-5
        config.training.early_stopping_patience = 15
        config.data.batch_size = 16
        config.data.augmentation_level = "medium"
        config.data.use_weighted_sampler = True
        config.training.use_mixup = True
        config.training.label_smoothing = 0.1
        return config
    
    @staticmethod
    def get_small_dataset_config() -> Config:
        """小数据集配置（强数据增强）"""
        config = Config()
        config.training.epochs = 150
        config.training.learning_rate = 1e-4
        config.data.batch_size = 8
        config.data.augmentation_level = "heavy"
        config.data.use_weighted_sampler = True
        config.training.use_mixup = True
        config.training.use_cutmix = True
        config.training.accumulate_grad_batches = 4
        return config
    
    @staticmethod
    def get_large_model_config() -> Config:
        """大模型配置"""
        config = Config()
        config.model.model_name = "efficientnet_b7"
        config.data.batch_size = 4
        config.training.accumulate_grad_batches = 8
        config.training.learning_rate = 5e-5
        config.data.cache_images = False  # 大模型通常内存紧张
        return config


# 使用示例
def create_config_example():
    """创建配置示例"""
    # 1. 使用默认配置
    config = Config()
    
    # 2. 自定义配置
    config.model.model_name = "convnext_tiny"
    config.data.batch_size = 32
    config.training.epochs = 50
    
    # 3. 使用模板
    fast_config = ConfigTemplates.get_fast_training_config()
    
    # 4. 保存配置
    config.save_config("config.json")
    
    return config


if __name__ == "__main__":
    config = create_config_example()
    print("✅ 配置创建完成")
    print(f"模型: {config.model.model_name}")
    print(f"输入尺寸: {config.data.input_size}")
    print(f"批次大小: {config.data.batch_size}")