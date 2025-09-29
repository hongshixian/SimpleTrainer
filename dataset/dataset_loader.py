import os
import json
from datasets import Dataset as HFDataset, load_dataset
from typing import Dict, Any, Optional, Union
from dataset.images.image_dataset import load_image_data_from_jsonl
from dataset.texts.text_dataset import load_text_data_from_jsonl
from dataset.audios.audio_dataset import load_audio_data_from_jsonl
from torchvision import transforms
from PIL import Image
from .custom_datasets import CUSTOM_DATASET_DICT
from .hf_datasets import HFDatasetWrapper


def load_image_dataset_from_jsonl(jsonl_path: str, transform=None) -> HFDataset:
    """
    从jsonl文件加载图像数据集
    :param jsonl_path: jsonl文件路径
    :param transform: 图像变换
    :return: HFDataset实例
    """
    return load_image_data_from_jsonl(jsonl_path, transform)


def load_text_dataset_from_jsonl(jsonl_path: str) -> HFDataset:
    """
    从jsonl文件加载文本数据集
    :param jsonl_path: jsonl文件路径
    :return: HFDataset实例
    """
    return load_text_data_from_jsonl(jsonl_path)


def load_audio_dataset_from_jsonl(jsonl_path: str) -> HFDataset:
    """
    从jsonl文件加载音频数据集
    :param jsonl_path: jsonl文件路径
    :return: HFDataset实例
    """
    return load_audio_data_from_jsonl(jsonl_path)


def get_jsonl_dataset(dataset_args: dict) -> dict:
    """
    根据配置获取数据集
    :param dataset_args: 数据集配置参数
    :return: 包含train_dataset和eval_dataset的字典
    """
    # 获取数据集类型
    dataset_type = dataset_args.get('dataset_type', 'image')
    
    # 获取训练和验证数据路径
    train_path = dataset_args.get('train_jsonline_path')
    val_path = dataset_args.get('val_jsonline_path')
    
    # 获取标签映射
    label2id = dataset_args.get('label2id', {})
    id2label = dataset_args.get('id2label', {})
    
    # 创建图像变换
    transform = None
    if dataset_type == 'image':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 加载训练数据集
    train_dataset = None
    if train_path and os.path.exists(train_path):
        if dataset_type == 'image':
            train_dataset = load_image_dataset_from_jsonl(train_path, transform)
        elif dataset_type == 'text':
            train_dataset = load_text_dataset_from_jsonl(train_path)
        elif dataset_type == 'audio':
            train_dataset = load_audio_dataset_from_jsonl(train_path)
    
    # 加载验证数据集
    eval_dataset = None
    if val_path and os.path.exists(val_path):
        if dataset_type == 'image':
            eval_dataset = load_image_dataset_from_jsonl(val_path, transform)
        elif dataset_type == 'text':
            eval_dataset = load_text_dataset_from_jsonl(val_path)
        elif dataset_type == 'audio':
            eval_dataset = load_audio_dataset_from_jsonl(val_path)
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset
    }


def get_hf_dataset(dataset_args: dict) -> dict:
    """
    从Hugging Face Hub加载数据集
    :param dataset_args: 数据集配置参数
    :return: 包含train_dataset和eval_dataset的字典，值为Hugging Face Dataset类型
    """
    # 获取数据集名称和配置
    hf_dataset_name = dataset_args.get('hf_dataset_name')
    hf_dataset_config = dataset_args.get('hf_dataset_config')
    
    # 获取训练和验证数据的split名称
    train_split = dataset_args.get('train_split', 'train')
    eval_split = dataset_args.get('eval_split', 'validation')
    
    # 加载训练数据集
    hf_train_dataset = load_dataset(hf_dataset_name, name=hf_dataset_config, split=train_split, trust_remote_code=True)
    train_dataset = HFDatasetWrapper(hf_train_dataset, transform=None)

    # 加载验证数据集
    if eval_split:
        hf_eval_dataset = load_dataset(hf_dataset_name, name=hf_dataset_config, split=eval_split, trust_remote_code=True)
        eval_dataset = HFDatasetWrapper(hf_eval_dataset, transform=None)
    else:
        eval_dataset = None
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset
    }


def get_custom_dataset(dataset_args: dict) -> dict:
    """
    根据配置获取自定义数据集
    :param dataset_args: 数据集配置参数
    :return: 包含train_dataset和eval_dataset的字典，值为Hugging Face Dataset类型
    """
    # 获取自定义数据集名称
    custom_dataset_name = dataset_args.get('custom_dataset_name')
    custom_dataset_args = dataset_args.get('custom_dataset_args', {})

    # 检查自定义数据集是否存在
    if custom_dataset_name not in CUSTOM_DATASET_DICT:
        raise ValueError(f"custom_dataset_name {custom_dataset_name} not found in CUSTOM_DATASET_DICT")
    custom_dataset_cls = CUSTOM_DATASET_DICT[custom_dataset_name]

    train_split = dataset_args.get('train_split')
    eval_split = dataset_args.get('eval_split')

    # 加载自定义数据集
    if train_split:
        train_dataset = custom_dataset_cls(split=train_split, **custom_dataset_args)
    else:
        train_dataset = None
        
    if eval_split:
        eval_dataset = custom_dataset_cls(split=eval_split, **custom_dataset_args)
    else:
        eval_dataset = None
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset
    }


def get_dataset(dataset_args: dict) -> dict:
    """
    根据配置获取数据集
    :param dataset_args: 数据集配置参数
    :return: 包含train_dataset和eval_dataset的字典，值为Hugging Face Dataset类型
    """
    # 检查是否指定了Hugging Face数据集名称
    hf_dataset_name = dataset_args.get('hf_dataset_name')
    if hf_dataset_name:
        return get_hf_dataset(dataset_args)
    
    # 加载custom数据集
    custom_dataset_name = dataset_args.get('custom_dataset_name')
    if custom_dataset_name:
        return get_custom_dataset(dataset_args)

    # 获取jsonl数据类型
    jsonl_dataset_type = dataset_args.get('dataset_type')
    if jsonl_dataset_type:
        return get_jsonl_dataset(dataset_args)
    
    raise KeyError(f"dataset args not avalible type : {dataset_args}")
