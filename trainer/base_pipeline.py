import torch
from transformers import Trainer, TrainingArguments
from utils.metric_utils import compute_metrics
from dataset.image_dataset import ImageDatasetFromJsonline
from dataset.text_dataset import TextDatasetFromJsonline
from utils.config_manager import ConfigManager
from utils.data_processor import DataProcessor


class BasePipeline:
    """
    所有Pipeline的基类，包含通用功能
    """
    
    def __init__(self, config_path):
        """
        初始化pipeline
        :param config_path: config.yaml文件路径
        """
        # 加载配置文件
        self.config_manager = ConfigManager(config_path)
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化数据集
        self.train_dataset, self.val_dataset = self._init_dataset()

        # 初始化数据处理器
        dataset_type = self.config_manager.get('dataset_type', 'image')
        self.data_processor = DataProcessor(dataset_type)

        # 初始化data collator
        self.data_collator = self._init_data_collator()
        
        # 初始化训练器
        self.trainer = self._init_trainer()
    

    
    def _init_model(self):
        """
        初始化模型，需要在子类中实现
        :return: 模型实例
        """
        raise NotImplementedError("_init_model method must be implemented in subclass")
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: (训练数据集实例, 验证数据集实例)
        """
        train_dataset = None
        val_dataset = None
        
        # 根据数据类型选择数据集类
        if self.config_manager.get('dataset_type', 'image') == 'image':
            # 从配置创建训练数据集
            if self.config_manager.get('train_jsonline_path', None):
                train_dataset = ImageDatasetFromJsonline(
                    jsonline_path=self.config_manager.get('train_jsonline_path', None)
                )
            
            # 从配置创建验证数据集
            if self.config_manager.get('val_jsonline_path', None):
                val_dataset = ImageDatasetFromJsonline(
                    jsonline_path=self.config_manager.get('val_jsonline_path', None)
                )
        elif self.config_manager.get('dataset_type', 'image') == 'text':
            # 从配置创建训练数据集
            if self.config_manager.get('train_jsonline_path', None):
                train_dataset = TextDatasetFromJsonline(
                    jsonline_path=self.config_manager.get('train_jsonline_path', None)
                )
            
            # 从配置创建验证数据集
            if self.config_manager.get('val_jsonline_path', None):
                val_dataset = TextDatasetFromJsonline(
                    jsonline_path=self.config_manager.get('val_jsonline_path', None)
                )
        
        return train_dataset, val_dataset

    def _init_data_collator(self):
        """
        初始化data collator，需要在子类中实现
        :return: DataCollator实例或function
        """
        raise NotImplementedError("_init_data_collator method must be implemented in subclass")
    
    def _init_trainer(self):
        """
        初始化训练器
        :return: Trainer实例
        """
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.config_manager.get('output_dir', './results'),
            num_train_epochs=self.config_manager.get('num_train_epochs', 3),
            per_device_train_batch_size=self.config_manager.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.config_manager.get('per_device_eval_batch_size', 8),
            warmup_steps=self.config_manager.get('warmup_steps', 500),
            weight_decay=self.config_manager.get('weight_decay', 0.01),
            logging_dir=self.config_manager.get('logging_dir', './logs'),
            logging_steps=self.config_manager.get('logging_steps', 10),
            save_steps=self.config_manager.get('save_steps', 1000),
            save_total_limit=self.config_manager.get('save_total_limit', 2),
            evaluation_strategy=self.config_manager.get('evaluation_strategy', 'steps'),
            eval_steps=self.config_manager.get('eval_steps', 1000),
            load_best_model_at_end=self.config_manager.get('load_best_model_at_end', True),
            report_to=self.config_manager.get('report_to', 'tensorboard'),
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,  # 使用验证数据集
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,  # 添加评估指标
        )
        
        return trainer
    
    def train(self):
        """
        训练模型
        """
        # 调用trainer的train方法完成训练
        self.trainer.train()
        
        # 保存模型和配置
        self.model.save_pretrained(self.config_manager.get('output_dir', './results'))
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(self.config_manager.get('output_dir', './results'))