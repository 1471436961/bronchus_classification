"""
项目常量定义
避免魔法数字，提高代码可维护性
"""

# 数据相关常量
DEFAULT_INPUT_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4
MAX_BATCH_SIZE = 256
MIN_BATCH_SIZE = 1

# 支持的图像格式
SUPPORTED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# 数据集划分默认比例
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# 模型相关常量
SUPPORTED_MODELS = [
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'
]

# 训练相关常量
DEFAULT_LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-8
MAX_LEARNING_RATE = 1.0
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 100
MIN_EPOCHS = 1
MAX_EPOCHS = 1000

# 优化器类型
SUPPORTED_OPTIMIZERS = ['adam', 'adamw', 'sgd', 'rmsprop']

# 学习率调度器类型
SUPPORTED_SCHEDULERS = ['cosine', 'step', 'exponential', 'plateau']

# 损失函数类型
SUPPORTED_LOSS_FUNCTIONS = ['crossentropy', 'focal', 'label_smoothing', 'arc_face']

# ArcFace相关常量
DEFAULT_ARC_MARGIN = 0.5
DEFAULT_ARC_SCALE = 30.0
MIN_ARC_MARGIN = 0.1
MAX_ARC_MARGIN = 1.0
MIN_ARC_SCALE = 1.0
MAX_ARC_SCALE = 100.0

# 数据增强级别
AUGMENTATION_LEVELS = ['none', 'light', 'medium', 'heavy']

# 注意力机制类型
SUPPORTED_ATTENTIONS = ['cbam', 'se', 'eca', 'none']

# 文件路径相关
DEFAULT_LOG_DIR = "../logs"
DEFAULT_WEIGHT_DIR = "../weight"
DEFAULT_DATA_DIR = "../data"
DEFAULT_CONFIG_FILE = "config.yaml"

# 模型保存相关
MODEL_SAVE_FORMAT = "model_epoch_{epoch:03d}_acc_{acc:.4f}.pth"
BEST_MODEL_NAME = "best_model.pth"
LAST_MODEL_NAME = "last_model.pth"

# 评估相关常量
TOP_K_ACCURACY = [1, 3, 5]
CONFUSION_MATRIX_FIGSIZE = (12, 10)
CLASSIFICATION_REPORT_DIGITS = 4

# 内存管理相关
MEMORY_CLEANUP_INTERVAL = 100  # 每100个batch清理一次内存
MAX_MEMORY_USAGE_RATIO = 0.9  # 最大内存使用比例

# 日志相关
LOG_INTERVAL = 10  # 每10个batch记录一次日志
SAVE_INTERVAL = 5  # 每5个epoch保存一次模型

# 早停相关
DEFAULT_PATIENCE = 10  # 早停耐心值
MIN_DELTA = 1e-4  # 最小改善阈值

# 数值稳定性相关
EPS = 1e-8  # 防止除零的小常数
CLIP_GRAD_NORM = 1.0  # 梯度裁剪阈值

# 随机种子
DEFAULT_SEED = 42

# 设备相关
AUTO_DEVICE = "auto"  # 自动选择设备
CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"