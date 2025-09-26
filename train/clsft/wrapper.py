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


def get_wrapper(model_args, device):
    finetuning_task = model_args.get("finetuning_task")
    if finetuning_task in ["audio-classification"]:
        return WrapperForAudioClassification(model_args, device)
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
