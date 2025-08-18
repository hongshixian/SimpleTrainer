import torch
import yaml
from transformers import Trainer, TrainingArguments
from pretrained_model.bert_classifier import BertForTextClassification, BertClassifierConfig
from dataset.text_dataset import TextDatasetFromJsonline
from utils.metric_utils import compute_metrics

class BertFinetunePipeline:
    """
    BERT分类器的训练pipeline类
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
        
        # 初始化数据集
        self.train_dataset, self.val_dataset = self._init_dataset()

        # 初始化训练器
        self.trainer = self._init_trainer()
    
    def _init_model(self):
        """
        初始化模型
        :return: BertForTextClassification模型实例
        """
        # 从配置创建模型配置
        model_config = BertClassifierConfig(
            model_name=self.config.get('model_name', 'bert-base-uncased'),
            label2id=self.config.get('label2id', None),
            id2label=self.config.get('id2label', None)
        )
        
        # 创建模型
        model = BertForTextClassification(model_config)
        return model
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: (训练数据集实例, 验证数据集实例)
        """
        # 从配置创建训练数据集
        train_dataset = TextDatasetFromJsonline(
            jsonline_path=self.config.get('train_jsonline_path', None),
            tokenizer=self.model.tokenizer,
            max_length=self.config.get('max_length', 512)
        )
        
        # 从配置创建验证数据集
        val_dataset = None
        if self.config.get('val_jsonline_path', None):
            val_dataset = TextDatasetFromJsonline(
                jsonline_path=self.config.get('val_jsonline_path', None),
                tokenizer=self.model.tokenizer,
                max_length=self.config.get('max_length', 512)
            )
        
        return train_dataset, val_dataset

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
            report_to=self.config.get('report_to', None),
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,  # 使用验证数据集
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
        self.model.save_pretrained(self.config.get('output_dir', './results'))
        self.model.config.save_pretrained(self.config.get('output_dir', './results'))