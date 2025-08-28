# 支气管分类项目使用指南

本指南将帮助您快速上手使用支气管分类项目。

## 🚀 快速开始

### 1. 环境验证
```bash
# 验证环境是否正确配置
python test_environment.py

# 运行单元测试
python run_tests.py
```

### 2. 数据准备
将您的数据按以下结构组织：
```
data/data_split/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 3. 开始训练
```bash
cd code
python train.py
```

### 4. 模型评估
```bash
cd code
python test.py
python test_advanced.py  # 高级评估功能
```

## 🔧 核心功能

### 统一日志系统
```python
from utils import setup_logger, get_logger

# 设置日志
setup_logger(level="INFO", log_file="../logs/training.log")

# 获取日志器
logger = get_logger(__name__)
logger.info("开始训练")
```

### 设备管理
```python
from utils import get_device, move_to_device, DeviceManager

# 自动选择最佳设备
device = get_device()

# 移动模型到设备
model = move_to_device(model, device)

# 使用设备管理器
dm = DeviceManager()
dm.clear_cache()  # 清理GPU缓存
```

### 错误处理
```python
from utils import handle_exceptions, ValidationError

@handle_exceptions(default_return=None)
def safe_training_step():
    # 训练代码
    pass

# 配置验证
try:
    config = DataConfig(batch_size=32)
except ValidationError as e:
    logger.error(f"配置错误: {e}")
```

### 性能监控
```python
from utils import profile_function, MemoryManager, GPUMemoryTracker

# 函数性能分析
@profile_function()
def training_epoch():
    pass

# 内存管理
memory_manager = MemoryManager()
memory_manager.cleanup_memory()

# GPU内存跟踪
with GPUMemoryTracker() as tracker:
    # 训练代码
    pass
```

## ⚙️ 配置系统

### 数据配置
```python
from config import DataConfig

config = DataConfig(
    batch_size=32,
    num_workers=4,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)
```

### 模型配置
```python
from config import ModelConfig

config = ModelConfig(
    model_name="efficientnet_b0",
    num_classes=33,
    pretrained=True,
    drop_rate=0.2
)
```

### 训练配置
```python
from config import TrainingConfig

config = TrainingConfig(
    epochs=100,
    learning_rate=1e-4,
    optimizer="adamw",
    scheduler="cosine",
    use_amp=True
)
```

## 📊 性能优化建议

### 1. 批次大小优化
```python
from utils import auto_adjust_batch_size

# 自动调整批次大小以适应GPU内存
optimal_batch_size = auto_adjust_batch_size(
    model=model,
    input_shape=(3, 224, 224),
    device=device,
    max_batch_size=128
)
```

### 2. 数据加载优化
```python
from utils import optimize_dataloader_workers

# 优化数据加载器worker数量
optimal_workers = optimize_dataloader_workers(
    dataset_size=len(dataset),
    batch_size=batch_size
)
```

### 3. 内存管理
```python
from utils import MemoryManager

# 在训练循环中使用内存管理
memory_manager = MemoryManager()

for epoch in range(epochs):
    for batch in dataloader:
        # 训练代码
        memory_manager.step()  # 定期清理内存
```

## 🧪 测试和验证

### 运行单元测试
```bash
# 运行所有测试
python run_tests.py

# 运行特定测试
python -m unittest tests.test_utils
python -m unittest tests.test_config
```

### 添加自定义测试
```python
import unittest
from utils import get_device

class TestCustomFunction(unittest.TestCase):
    def test_my_function(self):
        # 测试代码
        pass

if __name__ == '__main__':
    unittest.main()
```

## 🔍 调试和故障排除

### 1. 日志级别调整
```python
from utils import setup_logger

# 调试模式
setup_logger(level="DEBUG", log_file="../logs/debug.log")

# 生产模式
setup_logger(level="INFO", log_file="../logs/production.log")
```

### 2. 内存问题诊断
```python
from utils import GPUMemoryTracker, MemoryManager

# 跟踪内存使用
with GPUMemoryTracker() as tracker:
    # 问题代码
    pass

# 获取内存报告
memory_manager = MemoryManager()
print(memory_manager.get_memory_report())
```

### 3. 性能分析
```python
from utils import profile_function, global_monitor

@profile_function(include_memory=True)
def slow_function():
    # 需要分析的代码
    pass

# 获取性能报告
print(global_monitor.get_performance_report())
```

## 📈 最佳实践

### 1. 代码组织
- 使用新的工具模块进行统一管理
- 遵循项目的模块化结构
- 使用常量而非魔法数字

### 2. 错误处理
- 在关键函数中添加异常处理
- 使用项目定义的异常类型
- 记录详细的错误信息

### 3. 性能优化
- 定期清理GPU内存
- 使用性能分析装饰器
- 监控内存使用情况

### 4. 配置管理
- 使用配置类而非硬编码
- 验证配置参数的有效性
- 支持环境自适应配置

## 🔧 工具模块

### 核心工具
- `utils/device_utils.py` - 设备管理和GPU内存优化
- `utils/logger_utils.py` - 统一日志系统
- `utils/validation_utils.py` - 配置验证和环境检查
- `utils/error_handling.py` - 异常处理和错误恢复
- `utils/performance_utils.py` - 性能监控和优化
- `utils/constants.py` - 项目常量定义

### 配置系统
- 自动参数验证和类型检查
- 环境自适应配置
- 详细的错误提示和建议

### 测试框架
- 完整的单元测试覆盖
- 自动化测试运行
- 环境兼容性验证

### 依赖管理
- 精确版本锁定确保一致性
- 开发/生产环境分离
- 可选依赖灵活配置

## 📞 支持和帮助

如果您在使用过程中遇到问题：

1. 首先运行 `python test_environment.py` 验证环境配置
2. 检查 `logs/` 目录中的日志文件获取详细错误信息
3. 运行 `python run_tests.py` 确认核心功能正常
4. 查看代码注释和文档获取更多使用说明

项目具备生产级别的稳定性和可维护性，祝您使用愉快！