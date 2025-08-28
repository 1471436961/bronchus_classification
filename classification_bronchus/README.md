# 支气管分类系统

基于深度学习的支气管分类系统，支持多种先进的神经网络架构和完整的训练评估流程。

## ✨ 主要特性

- **多模型架构**: 支持EfficientNet、ResNet、ConvNeXt、Vision Transformer等主流模型
- **智能训练**: 混合精度训练、自动批次调整、学习率调度、早停机制
- **数据增强**: 多种数据增强策略，提升模型泛化能力
- **完整评估**: 多维度评估指标、混淆矩阵、ROC曲线、特征可视化
- **生产就绪**: 统一日志系统、错误处理、性能监控、自动化测试

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU训练推荐)

### 安装配置
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证环境
python test_environment.py

# 3. 运行测试
python run_tests.py
```

### 数据准备
```
data/data_split/
├── train/          # 训练数据
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/            # 验证数据
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/           # 测试数据
    ├── class1/
    ├── class2/
    └── ...
```

### 开始训练
```bash
cd code
python train.py
```

### 模型评估
```bash
cd code
python test.py              # 基础评估
python test_advanced.py     # 高级评估
```

## 🏗️ 项目架构

```
bronchus_classification/
├── code/                   # 核心代码
│   ├── model/             # 模型定义
│   ├── attentions/        # 注意力机制
│   ├── Blocks/            # 自定义模块
│   ├── loss/              # 损失函数
│   ├── utils/             # 工具模块
│   │   ├── device_utils.py      # 设备管理
│   │   ├── logger_utils.py      # 日志系统
│   │   ├── error_handling.py    # 错误处理
│   │   ├── performance_utils.py # 性能优化
│   │   └── validation_utils.py  # 配置验证
│   ├── config.py          # 配置管理
│   ├── train.py           # 训练脚本
│   └── test.py            # 评估脚本
├── data/                  # 数据目录
├── weight/                # 模型权重
├── logs/                  # 日志文件
└── tests/                 # 单元测试
```

## ⚙️ 配置系统

使用数据类进行配置管理，支持自动验证和环境适配：

```python
from config import DataConfig, ModelConfig, TrainingConfig

# 数据配置
data_config = DataConfig(
    batch_size=32,           # 自动验证范围
    num_workers=4,           # 自动检测CPU核心数
    train_ratio=0.7,         # 自动验证比例总和
    val_ratio=0.2,
    test_ratio=0.1
)

# 模型配置
model_config = ModelConfig(
    model_name="efficientnet_b0",
    num_classes=33,
    pretrained=True,
    drop_rate=0.2
)

# 训练配置
training_config = TrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    optimizer="adamw",
    scheduler="cosine",
    use_amp=True             # 混合精度训练
)
```

## 🛠️ 核心工具

### 设备管理
```python
from utils import get_device, move_to_device, DeviceManager

device = get_device()                    # 自动选择最佳设备
model = move_to_device(model, device)    # 智能对象移动
```

### 日志系统
```python
from utils import setup_logger, get_logger

setup_logger(level="INFO", log_file="../logs/training.log")
logger = get_logger(__name__)
```

### 性能监控
```python
from utils import profile_function, MemoryManager

@profile_function()
def training_step():
    pass

memory_manager = MemoryManager()
memory_manager.cleanup_memory()
```

### 错误处理
```python
from utils import handle_exceptions, ValidationError

@handle_exceptions(default_return=None)
def safe_operation():
    pass
```

## 📊 支持的模型

| 模型系列 | 变体 | 输入尺寸 | 参数量 |
|---------|------|----------|--------|
| EfficientNet | B0-B7 | 224-600 | 5M-66M |
| ResNet | 50/101/152 | 224 | 25M-60M |
| ConvNeXt | Tiny/Small/Base | 224 | 28M-89M |
| Vision Transformer | Base | 224 | 86M |
| Swin Transformer | Tiny | 224 | 28M |

## 📈 评估指标

- **分类指标**: 准确率、精确率、召回率、F1分数
- **可视化**: 混淆矩阵、ROC曲线、PR曲线
- **分析工具**: 特征可视化、错误样本分析
- **性能监控**: 训练曲线、内存使用、GPU利用率

## 🧪 质量保证

### 自动化测试
```bash
python run_tests.py         # 运行所有单元测试
python test_environment.py  # 环境兼容性测试
```

### 代码质量
- 统一的错误处理机制
- 完整的输入验证
- 自动化性能监控
- 内存泄漏检测

## 📚 文档

- [使用指南](USAGE_GUIDE.md) - 详细的使用说明和最佳实践
- 代码注释 - 完整的API文档和示例
- 单元测试 - 功能验证和使用示例

## 🔧 高级功能

### 自动优化
- 批次大小自动调整
- 数据加载器worker优化
- GPU内存自动管理

### 训练技巧
- 混合精度训练 (AMP)
- 梯度累积和裁剪
- 学习率预热和衰减
- 模型集成和蒸馏

### 部署支持
- 模型导出和转换
- 推理优化
- 批量预测
- 服务化部署

## 📄 许可证

MIT License - 详见 LICENSE 文件