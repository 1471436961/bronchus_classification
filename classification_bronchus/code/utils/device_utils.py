"""
设备管理工具
统一管理CUDA/CPU设备的使用
"""

import torch
from typing import Union, Optional, Any
from loguru import logger


class DeviceManager:
    """设备管理器，统一处理设备相关操作"""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        初始化设备管理器
        
        Args:
            device: 指定设备，None表示自动选择
        """
        if device is None:
            self.device = self._auto_select_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"设备管理器初始化完成，使用设备: {self.device}")
        
    def _auto_select_device(self) -> torch.device:
        """自动选择最佳设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"检测到CUDA设备: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("未检测到CUDA设备，使用CPU")
        return device
    
    def to_device(self, obj: Any) -> Any:
        """将对象移动到指定设备"""
        if hasattr(obj, 'to'):
            return obj.to(self.device)
        return obj
    
    def get_device(self) -> torch.device:
        """获取当前设备"""
        return self.device
    
    def is_cuda(self) -> bool:
        """检查是否使用CUDA"""
        return self.device.type == 'cuda'
    
    def get_memory_info(self) -> dict:
        """获取设备内存信息"""
        if self.is_cuda():
            return {
                'allocated': torch.cuda.memory_allocated(self.device),
                'cached': torch.cuda.memory_reserved(self.device),
                'max_allocated': torch.cuda.max_memory_allocated(self.device)
            }
        else:
            return {'message': 'CPU设备无内存统计'}


# 全局设备管理器实例
_device_manager = None


def get_device_manager() -> DeviceManager:
    """获取全局设备管理器实例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> torch.device:
    """获取当前设备"""
    return get_device_manager().get_device()


def move_to_device(obj: Any, device: Optional[torch.device] = None) -> Any:
    """将对象移动到指定设备"""
    if device is None:
        device = get_device()
    
    if hasattr(obj, 'to'):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA缓存已清理")