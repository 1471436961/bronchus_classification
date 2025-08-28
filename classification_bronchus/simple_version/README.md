# 支气管分类系统 - 简化版

这是支气管分类项目的简化版本，保留了核心功能，结构更加简洁，易于理解和使用。

## ✨ 特点

- **简洁架构**: 只包含核心功能，代码结构清晰
- **易于使用**: 一键训练、测试、预测
- **多模型支持**: EfficientNet、ResNet系列
- **完整功能**: 训练、验证、测试、预测、可视化

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
将数据按以下结构组织：
```
../data/data_split/
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

### 3. 配置参数
编辑 `config.py` 文件，设置模型和训练参数：
```python
class Config:
    MODEL_NAME = "efficientnet_b0"  # 模型选择
    NUM_CLASSES = 33               # 类别数量
    BATCH_SIZE = 32                # 批次大小
    NUM_EPOCHS = 100               # 训练轮数
    LEARNING_RATE = 1e-4           # 学习率
```

### 4. 开始训练
```bash
python train.py
```

### 5. 测试模型
```bash
python test.py
```

### 6. 预测新图像
```bash
# 单张图像预测
python predict.py --image path/to/image.jpg

# 批量预测
python predict.py --dir path/to/images/ --output results.csv
```

## 📁 项目结构

```
simple_version/
├── config.py          # 配置文件
├── dataset.py         # 数据处理
├── model.py           # 模型定义
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── predict.py         # 预测脚本
├── requirements.txt   # 依赖列表
└── README.md          # 说明文档
```

## ⚙️ 配置说明

### 支持的模型
- `efficientnet_b0` - EfficientNet-B0 (推荐)
- `resnet50` - ResNet-50
- `resnet101` - ResNet-101
- `resnet152` - ResNet-152

### 主要参数
- `MODEL_NAME`: 选择使用的模型
- `NUM_CLASSES`: 分类类别数量
- `BATCH_SIZE`: 训练批次大小
- `NUM_EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `IMAGE_SIZE`: 输入图像尺寸 (默认224)
- `USE_AUGMENTATION`: 是否使用数据增强

## 🔧 功能说明

### 训练 (train.py)
- 自动数据加载和预处理
- 支持数据增强
- 余弦退火学习率调度
- 自动保存最佳模型
- 实时显示训练进度
- 绘制训练曲线

### 测试 (test.py)
- 加载最佳模型进行测试
- 计算详细的分类指标
- 生成混淆矩阵
- 绘制各类别性能图表

### 预测 (predict.py)
- 单张图像预测
- 批量图像预测
- 输出置信度分数
- 支持CSV格式结果导出

## 📊 输出文件

训练和测试过程中会生成以下文件：

### 模型文件 (../weight/)
- `best_model.pth` - 最佳模型
- `latest_model.pth` - 最新模型

### 日志文件 (../logs/)
- `training_history.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `class_accuracy.png` - 各类别性能

## 🎯 使用示例

### 1. 基础训练
```bash
python train.py
```

### 2. 自定义配置训练
修改 `config.py` 后运行：
```bash
python train.py
```

### 3. 测试模型
```bash
python test.py
```

### 4. 预测单张图像
```bash
python predict.py --image sample.jpg --top_k 3
```

### 5. 批量预测
```bash
python predict.py --dir images/ --output predictions.csv
```

## 🔍 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `BATCH_SIZE`
   - 使用更小的模型 (如 efficientnet_b0)

2. **数据加载错误**
   - 检查数据目录结构
   - 确保图像文件格式正确

3. **模型加载失败**
   - 确保模型文件存在
   - 检查模型配置是否匹配

### 性能优化

1. **提高训练速度**
   - 增加 `NUM_WORKERS`
   - 使用更大的 `BATCH_SIZE`
   - 启用混合精度训练

2. **提高准确率**
   - 增加训练轮数
   - 调整学习率
   - 使用更大的模型
   - 增强数据增强策略

## 📄 许可证

MIT License

---

这个简化版本专注于核心功能，去除了复杂的工具模块，使代码更容易理解和修改。适合学习和快速原型开发使用。