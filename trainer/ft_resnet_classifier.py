import torch
from pretrained_model.resnet_classifier import ResNetForImageClassification, ResNetClassifierConfig
from trainer.base_pipeline import BasePipeline

class ResNetFinetunePipeline(BasePipeline):
    """
    ResNet分类器的训练pipeline类
    """
    
    def __init__(self, config_path):
        """
        初始化pipeline
        :param config_path: config.yaml文件路径
        """
        super().__init__(config_path)
    
    def _init_model(self):
        """
        初始化模型
        :return: ResNetForImageClassification模型实例
        """
        # 创建模型配置
        config = ResNetClassifierConfig(
            id2label=self.config_manager.get('id2label', None),
            label2id=self.config_manager.get('label2id', None),
            resnet_type=self.config_manager.get('resnet_type', 'resnet18'),
            num_classes=self.config_manager.get('num_classes', 1000)
        )
        
        # 创建模型
        model = ResNetForImageClassification(config)
        return model
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: (训练数据集实例, 验证数据集实例)
        """
        # 确保数据集类型为image
        if self.config_manager.get('dataset_type', 'image') != 'image':
            raise ValueError("ResNet Pipeline only supports image dataset type")
        return super()._init_dataset()

    def _init_data_collator(self):
        """
        初始化data collator
        :return: DataCollator实例或function
        """
        # 定义data collator函数
        def data_collator(features):
            # 使用数据处理器处理特征
            label2id = self.config_manager.get('label2id', {})
            return self.data_processor.process_features(
                features, 
                model=self.model, 
                dataset=self.train_dataset,
                label2id=label2id
            )
        
        return data_collator
    
    def _init_trainer(self):
        """
        初始化训练器
        :return: Trainer实例
        """
        return super()._init_trainer()
    
    def train(self):
        """
        训练模型
        """
        # 调用trainer的train方法完成训练
        self.trainer.train()
        
        # 保存模型和配置
        output_dir = self.config_manager.get('output_dir', './results')
        self.model.save_pretrained(output_dir)
        self.model.config.save_pretrained(output_dir)