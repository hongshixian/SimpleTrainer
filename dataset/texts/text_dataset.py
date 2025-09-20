import json
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


class TextDatasetFromJsonline(Dataset):
    """
    从jsonline文件加载文本数据集
    """
    def __init__(self, jsonline_path):
        """
        初始化数据集
        :param jsonline_path: jsonline文件路径
        """
        self.jsonline_path = jsonline_path
        self.data = []
        with open(jsonline_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.data.append(data)
        self.texts = [data['text'] for data in self.data]
        self.labels = [data['label'] for data in self.data]

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 索引
        :return: 包含文本和标签的字典
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 直接返回文本和标签
        return {
            'text': text,
            'label': label
        }


def load_text_data_from_jsonl(jsonline_path):
    """
    从jsonline文件加载文本数据并转换为Hugging Face Dataset格式
    :param jsonline_path: jsonline文件路径
    :return: Hugging Face Dataset实例
    """
    # 加载数据
    data = []
    with open(jsonline_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    # 创建Hugging Face Dataset
    hf_dataset = HFDataset.from_dict({
        'text': texts,
        'label': labels
    })
    
    return hf_dataset