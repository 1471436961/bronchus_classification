#!/usr/bin/env python3
"""
环境验证脚本 - 验证支气管分类项目环境是否配置正确
"""

import sys
import os
sys.path.append('./code')

def test_imports():
    """验证所有必要的包导入"""
    print("🔍 验证包导入...")
    
    try:
        import torch
        import torchvision
        import timm
        import torchmetrics
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import loguru
        import einops
        from torchsummary import summary
        print("✅ 所有必要的包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 包导入失败: {e}")
        return False

def test_pytorch():
    """验证PyTorch和CUDA"""
    print("\n🔍 验证PyTorch环境...")
    
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        
        # 验证简单的GPU操作
        try:
            x = torch.randn(2, 3).cuda()
            y = x * 2
            print("✅ GPU操作验证成功")
            return True
        except Exception as e:
            print(f"❌ GPU操作验证失败: {e}")
            return False
    else:
        print("⚠️  CUDA不可用，将使用CPU")
        return True

def test_config():
    """验证项目配置"""
    print("\n🔍 验证项目配置...")
    
    try:
        from config import Config
        config = Config()
        print(f"设备配置: {config.system.device}")
        print(f"分类数量: {config.model.num_classes}")
        print(f"训练数据路径: {config.data.train_path}")
        print(f"验证数据路径: {config.data.val_path}")
        print(f"数据路径: {config.data.test_path}")
        print(f"权重保存路径: {config.system.output_dir}")
        print("✅ 项目配置加载成功")
        return True
    except Exception as e:
        print(f"❌ 项目配置加载失败: {e}")
        return False

def test_model_loading():
    """验证模型加载"""
    print("\n🔍 验证模型加载...")
    
    try:
        import torch
        import timm
        from torchvision import models
        
        # 验证加载EfficientNet
        model = models.efficientnet_b0(pretrained=False)
        print("✅ EfficientNet-B0 加载成功")
        
        # 验证加载timm模型
        model_timm = timm.create_model('efficientnet_b0', pretrained=False)
        print("✅ TIMM EfficientNet-B0 加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_directories():
    """验证目录结构"""
    print("\n🔍 验证目录结构...")
    
    # 检查当前工作目录，决定使用相对路径还是绝对路径
    if os.path.basename(os.getcwd()) == 'code':
        # 在 code 目录下运行
        required_dirs = [
            '../data/data_split/train',
            '../data/data_split/val', 
            '../data/data_split/test',
            '../weight',
            '../logs'
        ]
    else:
        # 在项目根目录下运行
        required_dirs = [
            'data/data_split/train',
            'data/data_split/val', 
            'data/data_split/test',
            'weight',
            'logs'
        ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} 存在")
        else:
            print(f"❌ {dir_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主验证函数"""
    print("🚀 开始环境验证...\n")
    
    tests = [
        ("包导入", test_imports),
        ("PyTorch环境", test_pytorch),
        ("项目配置", test_config),
        ("模型加载", test_model_loading),
        ("目录结构", test_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}验证出现异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("📊 验证结果总结:")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项验证通过")
    
    if passed == len(results):
        print("\n🎉 恭喜！环境配置完全正确，可以开始使用项目了！")
        print("\n📝 使用提示:")
        print("1. 将您的数据放入 data/data_split/ 目录下的 train/val/test 文件夹")
        print("2. 运行 python train.py 开始训练")
        print("3. 运行 python test.py 进行评估")
        print("4. 运行 cd code && python test_advanced.py 进行高级评估")
    else:
        print(f"\n⚠️  有 {len(results) - passed} 项验证失败，请检查相关配置")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)