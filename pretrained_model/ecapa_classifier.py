import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Dict, Any
import librosa
import numpy as np
from network.ecapa_tdnn.model import ECAPA_TDNN
from .base_classifier import AudioClassifierBase


class ECAPAClassifierConfig(PretrainedConfig):
    model_type = "ecapa_classifier"
    
    def __init__(
        self,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        problem_type: Optional[str] = None,
        sampling_rate: int = 16000,
        channel_size: int = 1024,
        classifier_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label if id2label is not None else {}
        self.label2id = label2id if label2id is not None else {}
        self.problem_type = problem_type
        self.sampling_rate = sampling_rate
        # EPACA模型配置
        self.channel_size = channel_size
        # 分类头配置
        self.classifier_dropout = classifier_dropout

class ECAPAForAudioClassification(AudioClassifierBase):
    config_class = ECAPAClassifierConfig
    
    def __init__(self, config: ECAPAClassifierConfig):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        # 初始化ECAPA模型   
        self.model = ECAPA_TDNN(self.config.channel_size) 

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(192, config.num_labels)
        )
        
        # 初始化权重
        self.post_init()

    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对音频数据进行预处理
        :param examples: 包含音频数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        # 创建一个新的字典来存储预处理后的数据
        processed_examples = {}
        
        # 处理单个音频或批量音频
        if 'audio' in examples:
            audio_data = examples['audio']
        else:
            raise KeyError("No audio data found in examples. Available keys: {}".format(list(examples.keys())))
        
        # 转换为numpy数组
        audio_array = audio_data['array']
        sampling_rate = audio_data['sampling_rate']
        audio_array = audio_array.squeeze(0)

        # 转换为torch tensor
        processed_examples["input_values"] = audio_array

        # 处理标签
        processed_examples['labels'] = self.config.label2id[examples['label']]

        return processed_examples

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        
        return_dict = True  # 强制返回dict格式
        
        # 获取epaca输出
        feat = self.model(input_values, aug=False)
        # 分类头
        logits = self.classifier(feat)
        
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
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# 注册模型
AutoConfig.register("ecapa_classifier", ECAPAClassifierConfig)
AutoModel.register(ECAPAClassifierConfig, ECAPAForAudioClassification)
