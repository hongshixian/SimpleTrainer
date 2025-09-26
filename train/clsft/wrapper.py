import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Dict, Any
import numpy as np
from PIL import Image
from torchvision import transforms


def get_wrapper(model_args, device):
    finetuning_task = model_args.get("finetuning_task")
    if finetuning_task in ["audio-classification"]:
        return WrapperForAudioClassification(model_args, device)
    elif finetuning_task in ["image-classification"]:
        return WrapperForImageClassification(model_args, device)
    elif finetuning_task in ["text-classification"]:
        return WrapperForTextClassification(model_args, device)
    else:
        raise ValueError(f"finetuning_task {finetuning_task} is not supported")


class WrapperForAudioClassification:
    def __init__(self, model_args, device):
        finetuning_task = model_args.get("finetuning_task")
        if finetuning_task in ["audio-classification"]:
            self.processor = AutoFeatureExtractor.from_pretrained(**model_args)
            self.model = AutoModelForAudioClassification.from_pretrained(**model_args)
            self.model.to(device)
        else:
            raise ValueError(f"finetuning_task {finetuning_task} is not supported")

    def preprocess(self, examples: Dict[str, Any], is_train: bool = True):
        processed_examples = {}
        # 处理音频
        audio_array = examples["audio"]["array"]
        audio_sr = examples["audio"]["sampling_rate"]
        input_values = self.processor(
            audio_array, sampling_rate=audio_sr, return_tensors="pt")
        # 处理主要输入
        model_input_name = self.processor.model_input_names[0]
        processed_examples[model_input_name] = input_values[model_input_name].squeeze(0)
        # 处理标签
        if "label" in examples:
            label2id = self.model.config.label2id
            processed_examples["labels"] = label2id[examples["label"]]
        return processed_examples

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference method is not implemented")


class WrapperForImageClassification:
    def __init__(self, model_args, device):
        finetuning_task = model_args.get("finetuning_task")
        if finetuning_task in ["image-classification"]:
            # 对于图像分类，我们可能需要使用特定的特征提取器
            # 这里我们假设使用与ViT模型兼容的预处理
            self.processor = None  # 图像处理将在预处理函数中完成
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
            
            # 加载模型
            from transformers import AutoModelForImageClassification
            self.model = AutoModelForImageClassification.from_pretrained(**model_args)
            self.model.to(device)
        else:
            raise ValueError(f"finetuning_task {finetuning_task} is not supported")

    def preprocess(self, examples: Dict[str, Any], is_train: bool = True):
        processed_examples = {}
        # 获取变换方式
        transform = self.train_transform if is_train else self.eval_transform
        
        # 处理图像
        if 'image' in examples:
            image_data = examples['image']
        else:
            raise KeyError("No image data found in examples. Available keys: {}".format(list(examples.keys())))
        
        # 处理单个图像或批量图像
        if isinstance(image_data, list):
            # 批量处理
            images = [transform(image.convert('RGB')) for image in image_data]
            processed_examples['pixel_values'] = torch.stack(images)
        else:
            # 单个图像处理
            images = transform(image_data.convert('RGB'))
            processed_examples['pixel_values'] = images.unsqueeze(0)  # 添加batch维度

        # 处理标签
        if "label" in examples:
            label2id = self.model.config.label2id
            if isinstance(examples["label"], list):
                processed_examples["labels"] = torch.tensor([label2id[label] if isinstance(label, str) else label for label in examples["label"]], dtype=torch.long)
            else:
                label = examples["label"]
                processed_examples["labels"] = torch.tensor([label2id[label] if isinstance(label, str) else label], dtype=torch.long)
        
        return processed_examples

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference method is not implemented")


class WrapperForTextClassification:
    def __init__(self, model_args, device):
        finetuning_task = model_args.get("finetuning_task")
        if finetuning_task in ["text-classification"]:
            # 加载分词器
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(**model_args)
            # 加载模型
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(**model_args)
            self.model.to(device)
        else:
            raise ValueError(f"finetuning_task {finetuning_task} is not supported")

    def preprocess(self, examples: Dict[str, Any], is_train: bool = True):
        processed_examples = {}
        
        # 处理文本
        if 'text' in examples:
            text_data = examples['text']
        else:
            raise KeyError("No text data found in examples. Available keys: {}".format(list(examples.keys())))
        
        # 使用tokenizer处理文本
        processed = self.tokenizer(
            text_data,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 处理主要输入
        processed_examples['input_ids'] = processed.input_ids.squeeze(0) if isinstance(text_data, str) else processed.input_ids
        processed_examples['attention_mask'] = processed.attention_mask.squeeze(0) if isinstance(text_data, str) else processed.attention_mask

        # 处理标签
        if "label" in examples:
            label2id = self.model.config.label2id
            if isinstance(examples["label"], list):
                processed_examples["labels"] = torch.tensor([label2id[label] if isinstance(label, str) else label for label in examples["label"]], dtype=torch.long)
            else:
                label = examples["label"]
                processed_examples["labels"] = torch.tensor([label2id[label] if isinstance(label, str) else label], dtype=torch.long)
        
        return processed_examples

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs

    def inference(self, *args, **kwargs):
        raise NotImplementedError("inference method is not implemented")
