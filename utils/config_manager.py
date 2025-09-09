import yaml
import os

class ConfigManager:
    """
    配置管理器，用于加载和验证配置文件
    """
    
    def __init__(self, config_path):
        """
        初始化配置管理器
        :param config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path):
        """
        加载配置文件
        :param config_path: 配置文件路径
        :return: 配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key, default=None):
        """
        获取配置项的值
        :param key: 配置项键名
        :param default: 默认值
        :return: 配置项的值或默认值
        """
        return self.config.get(key, default)
    
    def validate_required_fields(self, required_fields):
        """
        验证必需的配置字段是否存在
        :param required_fields: 必需字段列表
        :raises ValueError: 如果有必需字段缺失
        """
        missing_fields = [field for field in required_fields if self.get(field) is None]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")
    
    def validate_pipeline_config(self):
        """
        验证Pipeline相关的配置
        :raises ValueError: 如果配置不合法
        """
        # 验证pipeline_type
        pipeline_type = self.get('pipeline_type')
        if pipeline_type not in ['clip_vit', 'bert', 'resnet']:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}. Must be one of ['clip_vit', 'bert', 'resnet']")
        
        # 验证数据集类型
        dataset_type = self.get('dataset_type', 'image')
        if dataset_type not in ['image', 'text']:
            raise ValueError(f"不支持的dataset_type: {dataset_type}")
        
        # 验证数据集路径
        if self.get('train_jsonline_path') is None:
            raise ValueError("train_jsonline_path is required")
        
        # 验证模型相关配置
        if pipeline_type == 'resnet':
            if self.get('resnet_type') is None:
                raise ValueError("resnet_type is required for resnet pipeline")
            if self.get('num_classes') is None:
                raise ValueError("num_classes is required for resnet pipeline")
        elif pipeline_type == 'bert':
            if self.get('model_name') is None:
                raise ValueError("model_name is required for bert pipeline")
        elif pipeline_type == 'clip_vit':
            if self.get('model_name') is None:
                raise ValueError("model_name is required for clip_vit pipeline")