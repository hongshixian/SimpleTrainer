import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Dict, Any
import librosa
import numpy as np
from network.aasist.AASIST import Model as AASISTModel
from .base_classifier import AudioClassifierBase


class AASISTClassifierConfig(PretrainedConfig):
    model_type = "aasist_classifier"
    
    def __init__(
        self,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        num_labels: int = 2,
        problem_type: Optional[str] = None,
        sampling_rate: int = 16000,
        d_args: dict = {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        },
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 分类相关配置
        self.id2label = id2label if id2label is not None else {}
        self.label2id = label2id if label2id is not None else {}
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.sampling_rate = sampling_rate
        # AASIST模型配置
        self.d_args = d_args
        #
        if not num_labels == 2:
            raise ValueError("AASISTClassifier only supports binary classification (num_labels=2).")


class AASISTForAudioClassification(AudioClassifierBase):
    config_class = AASISTClassifierConfig
    
    def __init__(self, config: AASISTClassifierConfig):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        # 初始化AASIST模型
        d_args = config.d_args
        self.model = AASISTModel(d_args)
        
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
        
        # 获取aasist输出
        hidden_states, output = self.model(input_values)
        logits = output
        
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
AutoConfig.register("aasist_classifier", AASISTClassifierConfig)
AutoModel.register(AASISTClassifierConfig, AASISTForAudioClassification)
