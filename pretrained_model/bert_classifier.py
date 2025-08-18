import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    AutoModel, 
    AutoTokenizer,
    AutoConfig
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union

class BertClassifierConfig(PretrainedConfig):
    model_type = "bert_classifier"
    
    def __init__(
        self,
        id2label: dict,
        label2id: dict,
        model_name: str = "bert-base-uncased",
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
        self.model_name = model_name


class BertForTextClassification(PreTrainedModel):
    config_class = BertClassifierConfig
    main_input_name = "input_ids"
    
    def __init__(self, config: BertClassifierConfig):
        super().__init__(config)
        
        self.num_classes = config.num_classes

        # 使用预训练的BERT模型
        self.bert = AutoModel.from_config(AutoConfig.from_pretrained(config.model_name))

        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.bert.config.hidden_size, config.num_classes)
        )
        
        # 初始化权重
        self.post_init()
    
    @classmethod
    def from_pretrained_model(
        cls,
        model_name: str,
        id2label: dict,
        label2id: dict,
        **kwargs
    ):
        """从预训练模型创建分类器"""
        
        # 加载预训练模型
        bert_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 创建配置
        config = BertClassifierConfig(
            id2label=id2label,
            label2id=label2id,
            model_name=model_name,
            **kwargs
        )
        
        # 创建模型
        model = cls(config)
        
        # 复制预训练权重
        model.bert.load_state_dict(bert_model.state_dict())
        
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[tuple, SequenceClassifierOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用池化后的输出 [batch_size, hidden_size]
        pooled_output = outputs.pooler_output
        
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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def freeze_bert(self):
        """冻结BERT模型的参数"""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert(self):
        """解冻BERT模型的参数"""
        for param in self.bert.parameters():
            param.requires_grad = True

# 注册模型（可选）
AutoConfig.register("bert_classifier", BertClassifierConfig)
AutoModel.register(BertClassifierConfig, BertForTextClassification)