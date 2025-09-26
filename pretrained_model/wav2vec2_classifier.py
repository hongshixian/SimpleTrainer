import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    Wav2Vec2Model,
    Wav2Vec2Processor
)
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Dict, Any
import librosa
import numpy as np
from .base_classifier import AudioClassifierBase


class Wav2Vec2ClassifierConfig(PretrainedConfig):
    model_type = "wav2vec2_classifier"
    
    def __init__(
        self,
        id2label: Optional[dict] = None,
        label2id: Optional[dict] = None,
        model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 10,
        classifier_dropout: float = 0.1,
        problem_type: Optional[str] = None,
        sampling_rate: int = 16000,
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
        self.sampling_rate = sampling_rate


class Wav2Vec2ForAudioClassification(AudioClassifierBase):
    config_class = Wav2Vec2ClassifierConfig
    
    def __init__(self, config: Wav2Vec2ClassifierConfig):
        super().__init__(config)
        
        self.num_labels = config.num_labels

        # 使用预训练的Wav2Vec2模型
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(config.classifier_dropout),
            nn.Linear(self.wav2vec2.config.hidden_size, config.num_labels)
        )
        
        # 初始化权重
        self.post_init()
        
        # 处理器用于音频预处理
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_name)
        
    @classmethod
    def from_pretrained_model(
        cls,
        model_name: str,
        id2label: dict,
        label2id: dict,
        num_labels: int = 10,
        **kwargs
    ):
        """从预训练模型创建分类器"""
        
        # 创建配置
        config = Wav2Vec2ClassifierConfig(
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
        
        # 如果是单个音频文件
        if isinstance(audio_data, dict) and 'array' in audio_data:
            # 处理音频数组
            audio_array = audio_data['array']
            sampling_rate = audio_data.get('sampling_rate', 16000)
            
            # 如果采样率不匹配，需要重采样
            if sampling_rate != self.config.sampling_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=self.config.sampling_rate)
            
            # 使用processor处理音频
            processed = self.processor(audio_array, sampling_rate=self.config.sampling_rate, return_tensors="pt")
            input_values = processed.input_values.squeeze(0)
            
        # 如果是批量音频
        elif isinstance(audio_data, list):
            input_values_list = []
            for audio_item in audio_data:
                if isinstance(audio_item, dict) and 'array' in audio_item:
                    audio_array = audio_item['array']
                    sampling_rate = audio_item.get('sampling_rate', 16000)
                    
                    # 如果采样率不匹配，需要重采样
                    if sampling_rate != self.config.sampling_rate:
                        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=self.config.sampling_rate)
                    
                    # 使用processor处理音频
                    processed = self.processor(audio_array, sampling_rate=self.config.sampling_rate, return_tensors="pt")
                    input_values_list.append(processed.input_values.squeeze(0))
                else:
                    raise ValueError("Invalid audio data format")
            
            # 对不同长度的音频进行填充
            if len(input_values_list) > 1:
                # 找到最大长度
                max_length = max([x.size(0) for x in input_values_list])
                # 对所有音频进行填充
                padded_input_values_list = []
                for input_values in input_values_list:
                    if input_values.size(0) < max_length:
                        # 使用零填充到最大长度
                        padding = torch.zeros(max_length - input_values.size(0))
                        padded_input_values = torch.cat([input_values, padding], dim=0)
                    else:
                        padded_input_values = input_values
                    padded_input_values_list.append(padded_input_values)
                # 堆叠批量数据
                input_values = torch.stack(padded_input_values_list, dim=0)
            else:
                # 只有一个音频样本
                input_values = torch.stack(input_values_list, dim=0)
        else:
            raise ValueError("Invalid audio data format")
            
        # 更新数据字典
        processed_examples['input_values'] = input_values

        # 处理标签
        if 'labels' in examples:
            processed_examples['labels'] = self.config.label2id[examples['labels']]
            processed_examples['label'] = self.config.label2id[examples['labels']]
        elif 'label' in examples:
            processed_examples['label'] = self.config.label2id[examples['label']]
        else:
            processed_examples['label'] = None
            raise ValueError("No label found in examples. Available keys: {}".format(list(examples.keys())))
        
        return processed_examples

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        
        return_dict = True  # 强制返回dict格式
        
        # 获取Wav2Vec2输出
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 使用时间维度上的平均池化
        pooled_output = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        
        # 通过分类头
        logits = self.classifier(pooled_output)
        
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

    def freeze_wav2vec2(self):
        """冻结Wav2Vec2的参数"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
    
    def unfreeze_wav2vec2(self):
        """解冻Wav2Vec2的参数"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True


# 注册模型
AutoConfig.register("wav2vec2_classifier", Wav2Vec2ClassifierConfig)
AutoModel.register(Wav2Vec2ClassifierConfig, Wav2Vec2ForAudioClassification)