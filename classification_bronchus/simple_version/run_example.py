"""
简化版运行示例
演示如何使用各个模块
"""

import os
import torch
from config import Config

def test_config():
    """测试配置"""
    print("=" * 60)
    print("1. 测试配置模块")
    print("=" * 60)
    
    Config.print_config()
    Config.create_dirs()
    
    print(f"✅ 配置测试完成")
    print(f"设备: {Config.DEVICE}")
    print(f"权重目录: {Config.WEIGHT_DIR}")
    print(f"日志目录: {Config.LOG_DIR}")

def test_dataset():
    """测试数据集"""
    print("\n" + "=" * 60)
    print("2. 测试数据集模块")
    print("=" * 60)
    
    try:
        from dataset import create_dataloaders
        train_loader, val_loader, test_loader, class_to_idx = create_dataloaders()
        
        print(f"✅ 数据集加载成功")
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        print(f"类别数量: {len(class_to_idx)}")
        
        # 测试一个批次
        if len(train_loader) > 0:
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"批次 {batch_idx}: 图像形状 {images.shape}, 标签形状 {labels.shape}")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        print("请确保数据目录存在并包含正确的数据结构")
        return False

def test_model():
    """测试模型"""
    print("\n" + "=" * 60)
    print("3. 测试模型模块")
    print("=" * 60)
    
    try:
        from model import create_model
        
        # 测试不同模型
        models_to_test = ["efficientnet_b0", "resnet50"]
        
        for model_name in models_to_test:
            print(f"\n测试模型: {model_name}")
            
            # 临时修改配置
            original_model = Config.MODEL_NAME
            Config.MODEL_NAME = model_name
            
            model = create_model()
            
            # 测试前向传播
            dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(Config.DEVICE)
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"✅ {model_name} 测试成功")
            print(f"输入形状: {dummy_input.shape}")
            print(f"输出形状: {output.shape}")
            
            # 恢复配置
            Config.MODEL_NAME = original_model
            
            # 清理内存
            del model
            if Config.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False

def test_training_setup():
    """测试训练设置"""
    print("\n" + "=" * 60)
    print("4. 测试训练设置")
    print("=" * 60)
    
    try:
        from train import Trainer
        
        # 创建训练器（但不开始训练）
        print("创建训练器...")
        trainer = Trainer()
        
        print(f"✅ 训练器创建成功")
        print(f"优化器: {type(trainer.optimizer).__name__}")
        print(f"调度器: {type(trainer.scheduler).__name__}")
        print(f"损失函数: {type(trainer.criterion).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练设置测试失败: {e}")
        return False

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 60)
    print("5. 使用示例")
    print("=" * 60)
    
    print("完整的使用流程:")
    print()
    print("1. 训练模型:")
    print("   python train.py")
    print()
    print("2. 测试模型:")
    print("   python test.py")
    print()
    print("3. 预测单张图像:")
    print("   python predict.py --image path/to/image.jpg")
    print()
    print("4. 批量预测:")
    print("   python predict.py --dir path/to/images/ --output results.csv")
    print()
    print("5. 自定义模型预测:")
    print("   python predict.py --image image.jpg --model path/to/model.pth")

def main():
    """主函数"""
    print("支气管分类系统 - 简化版测试")
    print("=" * 60)
    
    # 测试各个模块
    test_config()
    
    dataset_ok = test_dataset()
    if not dataset_ok:
        print("\n⚠️  数据集测试失败，但这不影响其他功能测试")
        print("如需完整测试，请准备数据后重新运行")
    
    model_ok = test_model()
    if not model_ok:
        print("\n❌ 模型测试失败，请检查依赖安装")
        return
    
    if dataset_ok:
        training_ok = test_training_setup()
        if not training_ok:
            print("\n❌ 训练设置测试失败")
    
    show_usage_examples()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    if dataset_ok and model_ok:
        print("✅ 所有核心功能正常，可以开始使用")
    else:
        print("⚠️  部分功能测试失败，请检查环境配置")

if __name__ == "__main__":
    main()