"""
配置验证工具
验证配置参数的有效性和路径的存在性
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger


class ValidationError(Exception):
    """配置验证异常"""
    pass


def validate_paths(paths: Dict[str, Union[str, Path]], create_missing: bool = False) -> Dict[str, bool]:
    """
    验证路径是否存在
    
    Args:
        paths: 路径字典 {name: path}
        create_missing: 是否创建缺失的目录
        
    Returns:
        验证结果字典 {name: exists}
        
    Raises:
        ValidationError: 路径验证失败
    """
    results = {}
    missing_paths = []
    
    for name, path in paths.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        results[name] = exists
        
        if not exists:
            missing_paths.append(f"{name}: {path}")
            if create_missing and not path_obj.suffix:  # 如果是目录
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    results[name] = True
                    logger.info(f"创建目录: {path}")
                except Exception as e:
                    logger.error(f"创建目录失败 {path}: {e}")
    
    if missing_paths and not create_missing:
        error_msg = f"以下路径不存在:\n" + "\n".join(missing_paths)
        logger.error(error_msg)
        raise ValidationError(error_msg)
    
    return results


def validate_config(config: Any) -> Dict[str, Any]:
    """
    验证配置对象的有效性
    
    Args:
        config: 配置对象
        
    Returns:
        验证结果字典
        
    Raises:
        ValidationError: 配置验证失败
    """
    validation_results = {
        'data_config': {},
        'model_config': {},
        'training_config': {},
        'paths': {},
        'errors': []
    }
    
    try:
        # 验证数据配置
        if hasattr(config, 'data'):
            validation_results['data_config'] = _validate_data_config(config.data)
        
        # 验证模型配置
        if hasattr(config, 'model'):
            validation_results['model_config'] = _validate_model_config(config.model)
        
        # 验证训练配置
        if hasattr(config, 'training'):
            validation_results['training_config'] = _validate_training_config(config.training)
        
        # 验证路径
        paths_to_check = {}
        if hasattr(config, 'data'):
            paths_to_check.update({
                'train_path': config.data.train_path,
                'val_path': config.data.val_path,
                'test_path': config.data.test_path
            })
        
        if paths_to_check:
            validation_results['paths'] = validate_paths(paths_to_check, create_missing=True)
        
    except Exception as e:
        validation_results['errors'].append(str(e))
        logger.error(f"配置验证失败: {e}")
    
    return validation_results


def _validate_data_config(data_config: Any) -> Dict[str, Any]:
    """验证数据配置"""
    results = {'valid': True, 'warnings': [], 'errors': []}
    
    # 检查批次大小
    if hasattr(data_config, 'batch_size'):
        if data_config.batch_size <= 0:
            results['errors'].append("batch_size必须大于0")
            results['valid'] = False
        elif data_config.batch_size > 128:
            results['warnings'].append("batch_size过大可能导致内存不足")
    
    # 检查数据集划分比例
    if hasattr(data_config, 'train_ratio') and hasattr(data_config, 'val_ratio') and hasattr(data_config, 'test_ratio'):
        total_ratio = data_config.train_ratio + data_config.val_ratio + data_config.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            results['errors'].append(f"数据集划分比例总和应为1.0，当前为{total_ratio}")
            results['valid'] = False
    
    # 检查输入尺寸
    if hasattr(data_config, 'input_size'):
        if not isinstance(data_config.input_size, (tuple, list)) or len(data_config.input_size) != 2:
            results['errors'].append("input_size应为长度为2的元组或列表")
            results['valid'] = False
    
    return results


def _validate_model_config(model_config: Any) -> Dict[str, Any]:
    """验证模型配置"""
    results = {'valid': True, 'warnings': [], 'errors': []}
    
    # 检查类别数量
    if hasattr(model_config, 'cls_num'):
        if model_config.cls_num <= 0:
            results['errors'].append("cls_num必须大于0")
            results['valid'] = False
    
    # 检查模型名称
    if hasattr(model_config, 'model_name'):
        supported_models = ['efficientnet', 'resnet', 'densenet', 'convnext']
        if model_config.model_name.lower() not in supported_models:
            results['warnings'].append(f"模型 {model_config.model_name} 可能不被支持")
    
    return results


def _validate_training_config(training_config: Any) -> Dict[str, Any]:
    """验证训练配置"""
    results = {'valid': True, 'warnings': [], 'errors': []}
    
    # 检查学习率
    if hasattr(training_config, 'lr'):
        if training_config.lr <= 0:
            results['errors'].append("学习率必须大于0")
            results['valid'] = False
        elif training_config.lr > 1.0:
            results['warnings'].append("学习率过大可能导致训练不稳定")
    
    # 检查训练轮数
    if hasattr(training_config, 'epochs'):
        if training_config.epochs <= 0:
            results['errors'].append("训练轮数必须大于0")
            results['valid'] = False
    
    return results


def validate_environment() -> Dict[str, Any]:
    """验证运行环境"""
    import torch
    import sys
    
    results = {
        'python_version': sys.version_info[:2],
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'warnings': [],
        'errors': []
    }
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        results['errors'].append("Python版本过低，建议使用3.8+")
    
    # 检查PyTorch版本
    pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    if pytorch_version < (1, 12):
        results['warnings'].append("PyTorch版本较低，建议升级到1.12+")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        results['warnings'].append("未检测到CUDA，将使用CPU训练（速度较慢）")
    
    return results