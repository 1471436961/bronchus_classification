"""
错误处理工具
提供统一的异常处理和错误恢复机制
"""

import functools
import traceback
from typing import Any, Callable, Optional, Type, Union
from loguru import logger


class BronchusClassificationError(Exception):
    """项目基础异常类"""
    pass


class DataError(BronchusClassificationError):
    """数据相关异常"""
    pass


class ModelError(BronchusClassificationError):
    """模型相关异常"""
    pass


class TrainingError(BronchusClassificationError):
    """训练相关异常"""
    pass


class ConfigError(BronchusClassificationError):
    """配置相关异常"""
    pass


def handle_exceptions(
    default_return: Any = None,
    exceptions: Union[Type[Exception], tuple] = Exception,
    reraise: bool = False,
    log_error: bool = True
):
    """
    异常处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        exceptions: 要捕获的异常类型
        reraise: 是否重新抛出异常
        log_error: 是否记录错误日志
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_error:
                    logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
                    logger.error(f"错误详情: {traceback.format_exc()}")
                
                if reraise:
                    raise
                
                return default_return
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_error: bool = True,
    **kwargs
) -> Any:
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或默认返回值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger.error(f"安全执行失败 {func.__name__}: {str(e)}")
        return default_return


def validate_input(
    validation_func: Callable,
    error_message: str = "输入验证失败"
):
    """
    输入验证装饰器
    
    Args:
        validation_func: 验证函数，返回True表示验证通过
        error_message: 验证失败时的错误消息
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validation_func(*args, **kwargs):
                raise ValueError(f"{error_message}: {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 延迟时间递增因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"函数 {func.__name__} 重试{max_retries}次后仍然失败: {str(e)}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 第{attempt + 1}次执行失败，{current_delay}秒后重试: {str(e)}")
                    
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
        return wrapper
    return decorator


class ErrorContext:
    """错误上下文管理器"""
    
    def __init__(
        self,
        error_message: str = "操作失败",
        reraise: bool = True,
        log_error: bool = True,
        cleanup_func: Optional[Callable] = None
    ):
        self.error_message = error_message
        self.reraise = reraise
        self.log_error = log_error
        self.cleanup_func = cleanup_func
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.log_error:
                logger.error(f"{self.error_message}: {str(exc_val)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
            
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"清理操作失败: {str(cleanup_error)}")
            
            if not self.reraise:
                return True  # 抑制异常
        
        return False


def log_function_call(include_args: bool = True, include_result: bool = False):
    """
    函数调用日志装饰器
    
    Args:
        include_args: 是否记录函数参数
        include_result: 是否记录函数返回值
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if include_args:
                args_str = f"args={args}, kwargs={kwargs}"
                logger.debug(f"调用函数 {func_name}({args_str})")
            else:
                logger.debug(f"调用函数 {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                if include_result:
                    logger.debug(f"函数 {func_name} 返回: {result}")
                else:
                    logger.debug(f"函数 {func_name} 执行完成")
                
                return result
            
            except Exception as e:
                logger.error(f"函数 {func_name} 执行失败: {str(e)}")
                raise
        
        return wrapper
    return decorator