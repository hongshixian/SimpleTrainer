import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, 
    PretrainedConfig
)
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union, Dict, Any
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import transforms
from PIL import Image


class ResNetClassifierConfig(PretrainedConfig):
    model_type = "resnet_classifier"
    
    def __init__(
        self,
        id2label: dict,
        label2id: dict,
        resnet_type: str = "resnet18",
        num_classes: int = 1000,
        classifier_dropout: float = 0.1,
        problem_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label
        self.label2id = label2id
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
        self.resnet_type = resnet_type


class ResNetForImageClassification(PreTrainedModel):
    config_class = ResNetClassifierConfig
    main_input_name = "pixel_values"
    
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
        # 获取变换方式
        transform = self.train_transform if is_train else self.eval_transform
        
        # 处理单个图像或批量图像
        if isinstance(examples['image'], list):
            # 批量处理
            images = [transform(image.convert('RGB')) for image in examples['image']]
        else:
            # 单个图像处理
            images = transform(examples['image'].convert('RGB'))
            
        # 更新数据字典
        examples['pixel_values'] = images

        # 将标签映射为ID
        if 'label' in examples:
            examples['label'] = [self.config.label2id[label] for label in examples['label']]
        
        return examples

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[tuple, ImageClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取ResNet输出
        logits = self.resnet(pixel_values)
        
        loss = None
        if labels is not None:
            # 根据问题类型计算损失
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
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