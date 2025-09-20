import json
import os
from datasets import Dataset as HFDataset
import librosa
import numpy as np


def load_audio_data_from_jsonl(jsonl_path: str) -> HFDataset:
    """
    从jsonl文件加载音频数据集
    :param jsonl_path: jsonl文件路径
    :return: HFDataset实例
    """
    data = {
        'audio': [],
        'label': []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # 加载音频文件
            if 'audio_path' in item:
                audio_path = item['audio_path']
                if os.path.exists(audio_path):
                    # 加载音频数据
                    audio_array, sampling_rate = librosa.load(audio_path, sr=None)
                    data['audio'].append({
                        'array': audio_array,
                        'sampling_rate': sampling_rate
                    })
                    # 添加标签
                    if 'label' in item:
                        data['label'].append(item['label'])
    
    # 创建Hugging Face Dataset
    dataset = HFDataset.from_dict(data)
    return dataset