import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    AutoConfig,
    AutoModelForImageClassification
)
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union, Dict, Any
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import transforms
from PIL import Image
from .base_classifier import ImageClassifierBase


class ResNetClassifierConfig(PretrainedConfig):
    model_type = "resnet_classifier"
    
    def __init__(
        self,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        resnet_type: str = "resnet18",
        num_classes: int = 1000,
        classifier_dropout: float = 0.1,
        problem_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label if id2label is not None else {}
        self.label2id = label2id if label2id is not None else {}
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
        self.resnet_type = resnet_type


class ResNetForImageClassification(ImageClassifierBase):
    config_class = ResNetClassifierConfig
    
    def __init__(self, config: ResNetClassifierConfig):
        super().__init__(config)
        
        self.num_classes = config.num_classes

        # 根据配置选择ResNet类型
        if config.resnet_type == "resnet18":
            self.resnet = resnet18(num_classes=config.num_classes)
        elif config.resnet_type == "resnet34":
            self.resnet = resnet34(num_classes=config.num_classes)
        elif config.resnet_type == "resnet50":
            self.resnet = resnet50(num_classes=config.num_classes)
        elif config.resnet_type == "resnet101":
            self.resnet = resnet101(num_classes=config.num_classes)
        elif config.resnet_type == "resnet152":
            self.resnet = resnet152(num_classes=config.num_classes)
        else:
            raise ValueError(f"Unsupported ResNet type: {config.resnet_type}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.num_classes, config.num_classes)
        )
        
        # 初始化权重
        self.post_init()
        
        # 定义训练和验证时的预处理变换
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def from_pretrained_model(
        cls,
        resnet_type: str,
        id2label: dict,
        label2id: dict,
        num_classes: int = 1000,
        **kwargs
    ):
        """从预训练模型创建分类器"""
        
        # 创建配置
        config = ResNetClassifierConfig(
            id2label=id2label,
            label2id=label2id,
            resnet_type=resnet_type,
            num_classes=num_classes,
            **kwargs
        )
        
        # 创建模型
        model = cls(config)
        
        return model


    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对图像数据进行预处理
        :param examples: 包含图像数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        # 创建一个新的字典来存储预处理后的数据
        processed_examples = {}
        
        # 获取变换方式
        transform = self.train_transform if is_train else self.eval_transform
        
        # 处理单个图像或批量图像
        # 确保我们使用正确的图像键
        if 'image' in examples:
            image_data = examples['image']
        elif 'images' in examples:
            image_data = examples['images']
        else:
            raise KeyError("No image data found in examples. Available keys: {}".format(list(examples.keys())))
        
        if isinstance(image_data, list):
            # 批量处理
            images = [transform(image.convert('RGB')) for image in image_data]
        else:
            # 单个图像处理
            images = transform(image_data.convert('RGB'))
            
        # 更新数据字典
        processed_examples['pixel_values'] = images

        # 将标签映射为ID
        if 'labels' in examples:
            # 处理单个标签或批量标签
            if isinstance(examples['labels'], list):
                processed_examples['labels'] = [self.config.label2id[label] if isinstance(label, str) else label for label in examples['labels']]
            else:
                processed_examples['labels'] = self.config.label2id[examples['labels']] if isinstance(examples['labels'], str) else examples['labels']
            # 重命名为label以匹配模型期望的输入
            processed_examples['label'] = processed_examples['labels']
        elif 'label' in examples:
            # 处理单个标签或批量标签
            if isinstance(examples['label'], list):
                processed_examples['label'] = [self.config.label2id[label] if isinstance(label, str) else label for label in examples['label']]
            else:
                processed_examples['label'] = self.config.label2id[examples['label']] if isinstance(examples['label'], str) else examples['label']
        
        return processed_examples

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> ImageClassifierOutput:
        
        return_dict = True  # 强制返回dict格式
        
        # 获取ResNet输出
        logits = self.resnet(pixel_values)
        
        loss = None
        if labels is not None:
            # 根据问题类型计算损失
            if self.config.problem_type is None:
                if self.config.num_classes == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_classes > 1:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def freeze_resnet(self):
        """冻结ResNet的参数"""
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def unfreeze_resnet(self):
        """解冻ResNet的参数"""
        for param in self.resnet.parameters():
            param.requires_grad = True


AutoConfig.register("resnet_classifier", ResNetClassifierConfig)
AutoModelForImageClassification.register(ResNetClassifierConfig, ResNetForImageClassification)
