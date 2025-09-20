import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoConfig,
    AutoModelForImageClassification,
    ViTModel
)
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union, Dict, Any
from torchvision import transforms
from PIL import Image


class ViTClassifierConfig(PretrainedConfig):
    model_type = "vit_classifier"
    
    def __init__(
        self,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 1000,
        classifier_dropout: float = 0.1,
        problem_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label if id2label is not None else {}
        self.label2id = label2id if label2id is not None else {}
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
        self.model_name = model_name


class ViTForImageClassification(PreTrainedModel):
    config_class = ViTClassifierConfig
    main_input_name = "pixel_values"
    
    def __init__(self, config: ViTClassifierConfig):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        # 使用预训练的ViT模型
        self.vit = ViTModel.from_pretrained(config.model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.vit.config.hidden_size, config.num_labels)
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
        model_name: str,
        id2label: dict,
        label2id: dict,
        num_labels: int = 1000,
        **kwargs
    ):
        """从预训练模型创建分类器"""
        
        # 创建配置
        config = ViTClassifierConfig(
            id2label=id2label,
            label2id=label2id,
            model_name=model_name,
            num_labels=num_labels,
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
    ) -> Union[tuple, ImageClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取ViT输出
        outputs = self.vit(pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 使用分类标记（通常是第一个标记）进行分类
        cls_token = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        
        # 通过分类头
        logits = self.classifier(cls_token)
        
        loss = None
        if labels is not None:
            # 根据问题类型计算损失
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
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

    def freeze_vit(self):
        """冻结ViT的参数"""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_vit(self):
        """解冻ViT的参数"""
        for param in self.vit.parameters():
            param.requires_grad = True


# 注册模型
AutoConfig.register("vit_classifier", ViTClassifierConfig)
AutoModelForImageClassification.register(ViTClassifierConfig, ViTForImageClassification)