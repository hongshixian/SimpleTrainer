import torch
import yaml
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from transformers import CLIPProcessor, CLIPModel
from pretrained_model.clip_vit_classifier import CLIPViTClassifier, CLIPViTClassifierConfig


class CLIPViTFinetunePipeline:
    """
    CLIP ViT分类器的训练pipeline类
    """
    
    def __init__(self, config_path):
        """
        初始化pipeline
        :param config_path: config.yaml文件路径
        """
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化处理器
        self.processor = self.model.processor
        
        # 初始化数据集
        self.dataset = self._init_dataset()
        
        # 初始化训练器
        self.trainer = self._init_trainer()
    
    def _init_model(self):
        """
        初始化模型
        :return: CLIPViTClassifier模型实例
        """
        # 从配置创建模型配置
        model_config = CLIPViTClassifierConfig.from_clip_pretrained(
            pretrained_model_name=self.config.get('model_name', 'openai/clip-vit-large-patch14'),
            label2id=self.config.get('label2id', None),
            id2label=self.config.get('id2label', None)
        )
        
        # 创建模型
        model = CLIPViTClassifier(model_config)
        return model
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: 数据集实例
        """
        # 这里应该根据配置文件中的数据集信息初始化数据集
        # 由于数据集的具体实现不在本任务范围内，这里返回None
        # 实际使用时需要根据配置文件中的数据集信息初始化数据集
        return None
    
    def _init_trainer(self):
        """
        初始化训练器
        :return: Trainer实例
        """
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', './results'),
            num_train_epochs=self.config.get('num_train_epochs', 3),
            per_device_train_batch_size=self.config.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 8),
            warmup_steps=self.config.get('warmup_steps', 500),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_dir=self.config.get('logging_dir', './logs'),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('save_steps', 1000),
            save_total_limit=self.config.get('save_total_limit', 2),
            evaluation_strategy=self.config.get('evaluation_strategy', 'steps'),
            eval_steps=self.config.get('eval_steps', 1000),
            load_best_model_at_end=self.config.get('load_best_model_at_end', True),
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            eval_dataset=self.dataset,  # 这里应该使用验证数据集
            # compute_metrics=compute_metrics,  # 如果需要评估指标
        )
        
        return trainer
    
    def train(self):
        """
        训练模型
        """
        # 调用trainer的train方法完成训练
        self.trainer.train()