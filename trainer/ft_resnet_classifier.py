import torch
import yaml
from transformers import Trainer, TrainingArguments
from pretrained_model.resnet_classifier import ResNetForImageClassification, ResNetClassifierConfig
from dataset.image_dataset import ImageDatasetFromJsonline
from utils.metric_utils import compute_metrics


class ResNetFinetunePipeline:
    """
    ResNet分类器的训练pipeline类
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

        # 初始化data collator
        self.data_collator = self._init_data_collator()
        
        # 初始化训练器
        self.trainer = self._init_trainer()
    
    def _init_model(self):
        """
        初始化模型
        :return: ResNetForImageClassification模型实例
        """
        # 创建模型配置
        config = ResNetClassifierConfig(
            id2label=self.config.get('id2label', None),
            label2id=self.config.get('label2id', None),
            resnet_type=self.config.get('resnet_type', 'resnet18'),
            num_classes=self.config.get('num_classes', 1000)
        )
        
        # 创建模型
        model = ResNetForImageClassification(config)
        return model
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: (训练数据集实例, 验证数据集实例)
        """
        # 从配置创建训练数据集
        train_dataset = ImageDatasetFromJsonline(
            jsonline_path=self.config.get('train_jsonline_path', None)
        )
        
        # 从配置创建验证数据集
        val_dataset = None
        if self.config.get('val_jsonline_path', None):
            val_dataset = ImageDatasetFromJsonline(
                jsonline_path=self.config.get('val_jsonline_path', None)
            )
        
        return train_dataset, val_dataset

    def _init_data_collator(self):
        """
        初始化data collator
        :return: DataCollator实例或function
        """
        # 定义data collator函数
        def data_collator(features):
            # 从features中提取图像和标签
            images = [f['image'] for f in features]
            labels = [f['label'] for f in features]
            
            # 应用transform处理图像
            # 注意：这里需要确保self.train_dataset.transform是一个可调用的transform函数
            # 如果没有定义transform，则需要定义一个默认的transform
            if self.train_dataset.transform is not None:
                pixel_values = torch.stack([self.train_dataset.transform(image) for image in images])
            else:
                # 如果没有transform，直接将PIL图像转换为tensor
                # 这里假设图像已经被预处理为相同大小
                pixel_values = torch.stack([torch.tensor(list(image.getdata())).view(3, image.size[1], image.size[0]).float() for image in images])
            
            # 使用model的label2id转换标签
            label_ids = torch.tensor([self.model.config.label2id[label] for label in labels])
            
            # 返回处理后的输入和标签
            return {
                'pixel_values': pixel_values,
                'labels': label_ids
            }
        
        return data_collator
    
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
        self.model.save_pretrained(self.config.get('output_dir', './results'))
        self.model.config.save_pretrained(self.config.get('output_dir', './results'))