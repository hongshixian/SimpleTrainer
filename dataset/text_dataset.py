import json
from torch.utils.data import Dataset


class TextDatasetFromJsonline(Dataset):
    """
    从jsonline文件加载文本数据集
    """
    def __init__(self, jsonline_path, tokenizer, max_length=512):
        """
        初始化数据集
        :param jsonline_path: jsonline文件路径
        :param tokenizer: 分词器
        :param max_length: 最大序列长度
        """
        self.jsonline_path = jsonline_path
        self.tokenizer = tokenizer
        self.max_length = max_length
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
        :return: 包含输入ID、注意力掩码和标签的字典
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 返回字典格式
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': label
        }