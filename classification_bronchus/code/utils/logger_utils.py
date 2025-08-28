"""
日志管理工具
统一配置和管理项目日志
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union
from loguru import logger


class LoggerConfig:
    """日志配置类"""
    
    # 日志级别
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    # 日志格式
    DEFAULT_FORMAT = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    SIMPLE_FORMAT = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )


def setup_logger(
    name: str = "bronchus_classification",
    level: str = LoggerConfig.INFO,
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_str: str = LoggerConfig.DEFAULT_FORMAT,
    console_output: bool = True
) -> None:
    """
    设置项目日志配置
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_str: 日志格式
        console_output: 是否输出到控制台
    """
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    if console_output:
        logger.add(
            sys.stdout,
            format=format_str,
            level=level,
            colorize=True
        )
    
    # 添加文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=format_str,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    logger.info(f"日志系统初始化完成 - 级别: {level}, 文件: {log_file}")


def get_logger(name: str = "bronchus_classification"):
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        logger实例
    """
    return logger.bind(name=name)


def setup_training_logger(log_dir: Union[str, Path] = "../logs"):
    """设置训练专用日志配置"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(
        name="training",
        level=LoggerConfig.INFO,
        log_file=log_dir / "training.log",
        format_str=LoggerConfig.DEFAULT_FORMAT
    )


def setup_evaluation_logger(log_dir: Union[str, Path] = "../logs"):
    """设置评估专用日志配置"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logger(
        name="evaluation",
        level=LoggerConfig.INFO,
        log_file=log_dir / "evaluation.log",
        format_str=LoggerConfig.DEFAULT_FORMAT
    )


def log_system_info():
    """记录系统信息"""
    import torch
    import platform
    
    logger.info("=== 系统信息 ===")
    logger.info(f"Python版本: {platform.python_version()}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info(f"操作系统: {platform.system()} {platform.release()}")
    logger.info("===============")


# 默认日志配置
def init_default_logger():
    """初始化默认日志配置"""
    setup_logger(
        log_file="../logs/app.log",
        level=LoggerConfig.INFO
    )
    log_system_info()