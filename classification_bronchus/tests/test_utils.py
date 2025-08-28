"""
工具模块单元测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from utils.device_utils import DeviceManager, get_device, move_to_device
from utils.validation_utils import validate_paths, ValidationError
from utils.constants import DEFAULT_BATCH_SIZE, SUPPORTED_MODELS


class TestDeviceUtils(unittest.TestCase):
    """设备工具测试"""
    
    def test_device_manager_init(self):
        """测试设备管理器初始化"""
        dm = DeviceManager()
        self.assertIsInstance(dm.get_device(), torch.device)
    
    def test_auto_device_selection(self):
        """测试自动设备选择"""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, ['cuda', 'cpu'])
    
    def test_move_to_device(self):
        """测试对象移动到设备"""
        tensor = torch.randn(2, 3)
        device = get_device()
        
        moved_tensor = move_to_device(tensor, device)
        self.assertEqual(moved_tensor.device.type, device.type)
    
    def test_move_list_to_device(self):
        """测试列表移动到设备"""
        tensors = [torch.randn(2, 3), torch.randn(3, 4)]
        device = get_device()
        
        moved_tensors = move_to_device(tensors, device)
        for tensor in moved_tensors:
            self.assertEqual(tensor.device.type, device.type)


class TestValidationUtils(unittest.TestCase):
    """验证工具测试"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_paths = {
            'existing_dir': self.temp_dir,
            'non_existing_dir': os.path.join(self.temp_dir, 'non_existing')
        }
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_existing_paths(self):
        """测试验证存在的路径"""
        paths = {'temp_dir': self.temp_dir}
        results = validate_paths(paths)
        self.assertTrue(results['temp_dir'])
    
    def test_validate_non_existing_paths(self):
        """测试验证不存在的路径"""
        non_existing_path = os.path.join(self.temp_dir, 'non_existing')
        paths = {'non_existing': non_existing_path}
        
        with self.assertRaises(ValidationError):
            validate_paths(paths)
    
    def test_create_missing_paths(self):
        """测试创建缺失的路径"""
        non_existing_path = os.path.join(self.temp_dir, 'new_dir')
        paths = {'new_dir': non_existing_path}
        
        results = validate_paths(paths, create_missing=True)
        self.assertTrue(results['new_dir'])
        self.assertTrue(os.path.exists(non_existing_path))


class TestConstants(unittest.TestCase):
    """常量测试"""
    
    def test_default_values(self):
        """测试默认值"""
        self.assertIsInstance(DEFAULT_BATCH_SIZE, int)
        self.assertGreater(DEFAULT_BATCH_SIZE, 0)
    
    def test_supported_models(self):
        """测试支持的模型列表"""
        self.assertIsInstance(SUPPORTED_MODELS, list)
        self.assertGreater(len(SUPPORTED_MODELS), 0)
        

        model_names = [model.lower() for model in SUPPORTED_MODELS]
        self.assertIn('efficientnet-b0', model_names)
        self.assertIn('resnet50', model_names)


if __name__ == '__main__':
    unittest.main()