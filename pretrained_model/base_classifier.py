import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Union, Dict, Any
from abc import ABC, abstractmethod


class BasePretrainedClassifier(PreTrainedModel, ABC):
    """所有预训练分类器的基类"""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        
    @abstractmethod
    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对输入数据进行预处理
        :param examples: 包含原始数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """前向传播"""
        pass
    
    @abstractmethod
    def inference(self, inputs) -> Dict[str, Any]:
        """
        对输入进行推理
        :param inputs: 输入数据
        :return: 包含标签和logits的字典
        """
        pass


class TextClassifierBase(BasePretrainedClassifier):
    """文本分类器基类"""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.main_input_name = "input_ids"
        
    @abstractmethod
    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对文本数据进行预处理
        :param examples: 包含文本数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        pass
    
    def inference(self, texts: list[str]) -> Dict[str, Any]:
        """
        对文本进行推理
        :param texts: 文本列表
        :return: 包含标签和logits的字典
        """
        with torch.no_grad():
            # 预处理
            processed = self.preprocess({"text": texts}, is_train=False)
            
            # 前向传播
            if "input_ids" in processed and "attention_mask" in processed:
                outputs = self.forward(
                    input_ids=processed["input_ids"],
                    attention_mask=processed["attention_mask"]
                )
            else:
                # 处理单个文本的情况
                outputs = self.forward(**processed)
            
            # 获取logits
            logits = outputs.logits
            
            # 转换为numpy数组
            logits_np = logits.cpu().numpy()
            
            # 获取预测的标签ID
            predicted_ids = np.argmax(logits_np, axis=-1)
            
            # 将ID转换为标签
            labels = []
            for idx in predicted_ids:
                if hasattr(self.config, 'id2label'):
                    labels.append(self.config.id2label.get(idx, f"unknown_{idx}"))
                else:
                    labels.append(f"unknown_{idx}")
            
            # 始终返回列表格式，保持API一致性
            return {
                "labels": labels,
                "logits": logits_np
            }


class ImageClassifierBase(BasePretrainedClassifier):
    """图像分类器基类"""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.main_input_name = "pixel_values"
        
    @abstractmethod
    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对图像数据进行预处理
        :param examples: 包含图像数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        pass
    
    def inference(self, images: list) -> Dict[str, Any]:
        """
        对图像进行推理
        :param images: PIL图像列表
        :return: 包含标签和logits的字典
        """
        with torch.no_grad():
            # 预处理
            processed = self.preprocess({"image": images}, is_train=False)
            
            # 前向传播
            if "pixel_values" in processed:
                outputs = self.forward(pixel_values=processed["pixel_values"])
            else:
                outputs = self.forward(**processed)
            
            # 获取logits
            logits = outputs.logits
            
            # 转换为numpy数组
            logits_np = logits.cpu().numpy()
            
            # 获取预测的标签ID
            predicted_ids = np.argmax(logits_np, axis=-1)
            
            # 将ID转换为标签
            labels = []
            for idx in predicted_ids:
                if hasattr(self.config, 'id2label'):
                    labels.append(self.config.id2label.get(idx, f"unknown_{idx}"))
                else:
                    labels.append(f"unknown_{idx}")
            
            # 始终返回列表格式，保持API一致性
            return {
                "labels": labels,
                "logits": logits_np
            }


class AudioClassifierBase(BasePretrainedClassifier):
    """音频分类器基类"""
    
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.main_input_name = "input_values"
        
    @abstractmethod
    def preprocess(self, examples: Dict[str, Any], is_train: bool = False) -> Dict[str, Any]:
        """
        对音频数据进行预处理
        :param examples: 包含音频数据的字典
        :param is_train: 是否为训练模式
        :return: 预处理后的数据字典
        """
        pass
    
    def inference(self, audios: list) -> Dict[str, Any]:
        """
        对音频进行推理
        :param audios: 音频数据列表
        :return: 包含标签和logits的字典
        """
        with torch.no_grad():
            # 预处理
            processed = self.preprocess({"audio": audios}, is_train=False)
            
            # 前向传播
            if "input_values" in processed:
                outputs = self.forward(input_values=processed["input_values"])
            else:
                outputs = self.forward(**processed)
            
            # 获取logits
            logits = outputs.logits
            
            # 转换为numpy数组
            logits_np = logits.cpu().numpy()
            
            # 获取预测的标签ID
            predicted_ids = np.argmax(logits_np, axis=-1)
            
            # 将ID转换为标签
            labels = []
            for idx in predicted_ids:
                if hasattr(self.config, 'id2label'):
                    labels.append(self.config.id2label.get(idx, f"unknown_{idx}"))
                else:
                    labels.append(f"unknown_{idx}")
            
            # 始终返回列表格式，保持API一致性
            return {
                "labels": labels,
                "logits": logits_np
            }