import torch
from pretrained_model.clip_vit_classifier import CLIPViTClassifier, CLIPViTClassifierConfig
from trainer.base_pipeline import BasePipeline

class CLIPViTFinetunePipeline(BasePipeline):
    """
    CLIP ViT分类器的训练pipeline类
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
        :return: CLIPViTClassifier模型实例
        """
        # 从配置创建模型配置
        model_config = CLIPViTClassifierConfig.from_clip_pretrained(
            pretrained_model_name=self.config_manager.get('model_name', 'openai/clip-vit-large-patch14'),
            label2id=self.config_manager.get('label2id', None),
            id2label=self.config_manager.get('id2label', None)
        )
        
        # 创建模型
        model = CLIPViTClassifier(model_config)
        return model
    
    def _init_dataset(self):
        """
        初始化数据集
        :return: (训练数据集实例, 验证数据集实例)
        """
        # 确保数据集类型为image
        if self.config_manager.get('dataset_type', 'image') != 'image':
            raise ValueError("CLIP ViT Pipeline only supports image dataset type")
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