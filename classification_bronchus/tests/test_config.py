"""
配置模块单元测试
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from config import DataConfig, ModelConfig, TrainingConfig
from utils.validation_utils import ValidationError


class TestDataConfig(unittest.TestCase):
    """数据配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DataConfig()
        self.assertIsInstance(config.batch_size, int)
        self.assertGreater(config.batch_size, 0)

        self.assertIsInstance(config.num_workers, int)
    
    def test_invalid_batch_size(self):
        """测试无效批次大小"""
        try:
            config = DataConfig(batch_size=0)
            self.fail("应该抛出ValidationError")
        except ValidationError:
            pass
        
        try:
            config = DataConfig(batch_size=1000)
            self.fail("应该抛出ValidationError")
        except ValidationError:
            pass
    
    def test_invalid_ratio_sum(self):
        """测试无效的比例总和"""
        try:
            config = DataConfig(train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)
            self.fail("应该抛出ValidationError")
        except ValidationError:
            pass
    
    def test_valid_ratios(self):
        """测试有效的比例设置"""
        config = DataConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        self.assertAlmostEqual(
            config.train_ratio + config.val_ratio + config.test_ratio, 
            1.0, 
            places=6
        )


class TestModelConfig(unittest.TestCase):
    """模型配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        self.assertIsInstance(config.num_classes, int)
        self.assertGreater(config.num_classes, 0)
    
    def test_model_selection(self):
        """测试模型选择"""
        config = ModelConfig(model_name='efficientnet-b0')
        self.assertEqual(config.model_name, 'efficientnet-b0')


class TestTrainingConfig(unittest.TestCase):
    """训练配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        self.assertIsInstance(config.learning_rate, float)
        self.assertGreater(config.learning_rate, 0)
        self.assertIsInstance(config.epochs, int)
        self.assertGreater(config.epochs, 0)
    
    def test_learning_rate_bounds(self):
        """测试学习率边界"""
        # 测试有效学习率
        config = TrainingConfig(learning_rate=1e-4)
        self.assertEqual(config.learning_rate, 1e-4)
        
        # 测试边界值
        config = TrainingConfig(learning_rate=1e-8)  # 最小值
        self.assertEqual(config.learning_rate, 1e-8)


if __name__ == '__main__':
    unittest.main()