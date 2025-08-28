"""
性能优化工具
提供内存管理、性能监控等功能
"""

import gc
import time
import psutil
import functools
from typing import Any, Callable, Dict, Optional
import torch
from loguru import logger
from .constants import MEMORY_CLEANUP_INTERVAL, MAX_MEMORY_USAGE_RATIO


class MemoryManager:
    """内存管理器"""
    
    def __init__(self, cleanup_interval: int = MEMORY_CLEANUP_INTERVAL):
        self.cleanup_interval = cleanup_interval
        self.step_count = 0
        self.initial_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved() / (1024**3),  # GB
                'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
            })
        
        return memory_info
    
    def cleanup_memory(self, force: bool = False):
        """清理内存"""
        if force or self.step_count % self.cleanup_interval == 0:
            # Python垃圾回收
            gc.collect()
            
            # CUDA内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            current_memory = self._get_memory_usage()
            logger.debug(f"内存清理完成 - 系统内存: {current_memory['system_memory_percent']:.1f}%")
            
            if torch.cuda.is_available():
                logger.debug(f"GPU内存: {current_memory['gpu_memory_allocated']:.2f}GB")
    
    def step(self):
        """步进计数器"""
        self.step_count += 1
        self.cleanup_memory()
    
    def check_memory_usage(self) -> bool:
        """检查内存使用是否超限"""
        memory_info = self._get_memory_usage()
        
        # 检查系统内存
        if memory_info['system_memory_percent'] > MAX_MEMORY_USAGE_RATIO * 100:
            logger.warning(f"系统内存使用过高: {memory_info['system_memory_percent']:.1f}%")
            return False
        
        # 检查GPU内存
        if torch.cuda.is_available() and memory_info['gpu_memory_percent'] > MAX_MEMORY_USAGE_RATIO * 100:
            logger.warning(f"GPU内存使用过高: {memory_info['gpu_memory_percent']:.1f}%")
            return False
        
        return True
    
    def get_memory_report(self) -> str:
        """获取内存使用报告"""
        current_memory = self._get_memory_usage()
        
        report = f"""
内存使用报告:
- 系统内存使用: {current_memory['system_memory_percent']:.1f}%
- 系统可用内存: {current_memory['system_memory_available']:.2f}GB
"""
        
        if torch.cuda.is_available():
            report += f"""- GPU内存已分配: {current_memory['gpu_memory_allocated']:.2f}GB
- GPU内存已保留: {current_memory['gpu_memory_reserved']:.2f}GB
- GPU内存使用率: {current_memory['gpu_memory_percent']:.1f}%
"""
        
        return report


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.timers = {}
        self.counters = {}
        self.memory_manager = MemoryManager()
    
    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name not in self.timers:
            logger.warning(f"计时器 {name} 未启动")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """增加计数器"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def get_counter(self, name: str) -> int:
        """获取计数器值"""
        return self.counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """重置计数器"""
        self.counters[name] = 0
    
    def get_performance_report(self) -> str:
        """获取性能报告"""
        report = "性能监控报告:\n"
        
        if self.counters:
            report += "计数器:\n"
            for name, value in self.counters.items():
                report += f"  {name}: {value}\n"
        
        report += self.memory_manager.get_memory_report()
        
        return report


def profile_function(include_memory: bool = True):
    """函数性能分析装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # 记录开始时间和内存
            start_time = time.time()
            if include_memory:
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            try:
                result = func(*args, **kwargs)
                
                # 记录结束时间和内存
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                log_msg = f"函数 {func_name} 执行时间: {elapsed_time:.4f}秒"
                
                if include_memory and torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    memory_diff = (end_memory - start_memory) / (1024**2)  # MB
                    log_msg += f", 内存变化: {memory_diff:+.2f}MB"
                
                logger.debug(log_msg)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.error(f"函数 {func_name} 执行失败 (耗时: {elapsed_time:.4f}秒): {str(e)}")
                raise
        
        return wrapper
    return decorator


def optimize_dataloader_workers(dataset_size: int, batch_size: int) -> int:
    """优化数据加载器的worker数量"""
    # 基于数据集大小和批次大小动态调整
    cpu_count = psutil.cpu_count()
    
    # 计算每个worker处理的样本数
    samples_per_worker = dataset_size / cpu_count
    
    # 如果每个worker处理的样本太少，减少worker数量
    if samples_per_worker < batch_size * 2:
        optimal_workers = max(1, dataset_size // (batch_size * 2))
    else:
        optimal_workers = min(cpu_count, 8)  # 限制最大worker数量
    
    logger.info(f"优化数据加载器worker数量: {optimal_workers} (CPU核心数: {cpu_count})")
    return optimal_workers


def auto_adjust_batch_size(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    max_batch_size: int = 256,
    min_batch_size: int = 1
) -> int:
    """自动调整批次大小以适应GPU内存"""
    if not torch.cuda.is_available():
        return min_batch_size
    
    model.eval()
    batch_size = max_batch_size
    
    while batch_size >= min_batch_size:
        try:
            # 创建测试输入
            test_input = torch.randn(batch_size, *input_shape, device=device)
            
            # 测试前向传播
            with torch.no_grad():
                _ = model(test_input)
            
            # 清理测试数据
            del test_input
            torch.cuda.empty_cache()
            
            logger.info(f"自动调整批次大小: {batch_size}")
            return batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                torch.cuda.empty_cache()
                logger.warning(f"批次大小 {batch_size * 2} 导致内存不足，尝试 {batch_size}")
            else:
                raise e
    
    logger.warning(f"无法找到合适的批次大小，使用最小值: {min_batch_size}")
    return min_batch_size


class GPUMemoryTracker:
    """GPU内存跟踪器"""
    
    def __init__(self):
        self.peak_memory = 0
        self.start_memory = 0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            
            logger.info(f"GPU内存使用情况:")
            logger.info(f"  开始: {self.start_memory / (1024**2):.2f}MB")
            logger.info(f"  峰值: {self.peak_memory / (1024**2):.2f}MB")
            logger.info(f"  结束: {current_memory / (1024**2):.2f}MB")
            logger.info(f"  净增长: {(current_memory - self.start_memory) / (1024**2):+.2f}MB")


# 全局性能监控器实例
global_monitor = PerformanceMonitor()