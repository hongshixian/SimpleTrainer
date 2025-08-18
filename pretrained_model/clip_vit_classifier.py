import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    CLIPVisionModel, 
    CLIPVisionConfig,
    CLIPImageProcessor,
    AutoConfig,
    AutoModelForImageClassification
)
from transformers.modeling_outputs import ImageClassifierOutput
from typing import Optional, Union

# 1. 定义配置类
class CLIPViTClassifierConfig(PretrainedConfig):
    model_type = "clip_vit_classifier"
    
    def __init__(
        self,
        id2label: dict,
        label2id: dict,
        vision_config: Optional[dict] = None,
        processor_config: Optional[dict] = None,
        classifier_dropout: float = 0.1,
        problem_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label
        self.label2id = label2id
        self.num_classes = len(id2label)
        self.classifier_dropout = classifier_dropout
        self.problem_type = problem_type
        
        # CLIP Vision配置
        if vision_config is None:
            vision_config = {
                "hidden_size": 1024,  # ViT-Large
                "image_size": 336,
                "patch_size": 14,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "hidden_act": "quick_gelu",
                "layer_norm_eps": 1e-5,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "initializer_range": 0.02,
                "initializer_factor": 1.0,
            }
        
        self.vision_config = CLIPVisionConfig(**vision_config)

        # CLIPImageProcessor 配置
        if processor_config is None:
            processor_config = {
                "image_size": 336,
                "resample": 3,
                "do_normalize": True,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
            }
        self.processor_config = CLIPImageProcessor(**processor_config)

# 2. 定义模型类
class CLIPViTForImageClassification(PreTrainedModel):
    config_class = CLIPViTClassifierConfig
    main_input_name = "pixel_values"
    
    def __init__(self, config: CLIPViTClassifierConfig):
        super().__init__(config)
        
        self.num_classes = config.num_classes

        # 使用CLIP的视觉编码器
        self.vision_model = CLIPVisionModel(config.vision_config)

        # 图像处理器
        self.processor = CLIPImageProcessor(config.processor_config)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.vision_config.hidden_size, config.num_classes)
        )
        
        # 初始化权重
        self.post_init()
    
    @classmethod
    def from_clip_pretrained(
        cls,
        clip_model_name: str,
        id2label: dict,
        label2id: dict,
        **kwargs
    ):
        """从预训练CLIP模型创建分类器"""
        
        # 加载预训练CLIP模型的视觉部分
        clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        # 创建配置
        vision_config = clip_model.config.vision_config.to_dict()
        processor_config = clip_processor.to_dict()
        
        config = CLIPViTClassifierConfig(
            id2label=id2label,
            label2id=label2id,
            vision_config=vision_config,
            processor_config=processor_config,
            **kwargs
        )
        
        # 创建模型
        model = cls(config)
        
        # 复制预训练权重
        model.vision_model.load_state_dict(clip_model.vision_model.state_dict())
        
        return model
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[tuple, ImageClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取视觉特征
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )
        
        # 使用池化后的输出 [batch_size, hidden_size]
        pooled_output = vision_outputs.pooler_output
        
        # 分类
        logits = self.classifier(pooled_output)
        
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
            output = (logits,) + vision_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )
    
    def freeze_vision_model(self):
        """冻结视觉编码器的参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
    
    def unfreeze_vision_model(self):
        """解冻视觉编码器的参数"""
        for param in self.vision_model.parameters():
            param.requires_grad = True

# 3. 注册模型（可选）
AutoConfig.register("clip_vit_classifier", CLIPViTClassifierConfig)
AutoModelForImageClassification.register(CLIPViTClassifierConfig, CLIPViTForImageClassification)