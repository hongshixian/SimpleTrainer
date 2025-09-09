import torch


class DataProcessor:
    """
    数据处理器，用于处理不同类型的数据
    """
    
    def __init__(self, dataset_type='image'):
        """
        初始化数据处理器
        :param dataset_type: 数据集类型 ('image' 或 'text')
        """
        self.dataset_type = dataset_type
    
    def process_features(self, features, model=None, dataset=None, label2id=None):
        """
        处理特征数据
        :param features: 特征列表
        :param model: 模型实例（可选）
        :param dataset: 数据集实例（可选）
        :param label2id: 标签到ID的映射（可选）
        :return: 处理后的数据字典
        """
        if self.dataset_type == 'image':
            return self._process_image_features(features, model, dataset, label2id)
        elif self.dataset_type == 'text':
            return self._process_text_features(features, model, dataset, label2id)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _process_image_features(self, features, model=None, dataset=None, label2id=None):
        """
        处理图像特征数据
        :param features: 图像特征列表
        :param model: 模型实例（可选）
        :param dataset: 数据集实例（可选）
        :param label2id: 标签到ID的映射（可选）
        :return: 处理后的数据字典
        """
        # 从features中提取图像和标签
        images = [f['image'] for f in features]
        labels = [f['label'] for f in features]
        
        # 处理图像
        if model and hasattr(model, 'processor'):
            # 使用模型的processor处理图像
            inputs = model.processor(
                images=images,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            pixel_values = inputs['pixel_values']
        elif dataset and hasattr(dataset, 'transform') and dataset.transform is not None:
            # 使用数据集的transform处理图像
            pixel_values = torch.stack([dataset.transform(image) for image in images])
        else:
            # 默认处理方式：直接将PIL图像转换为tensor
            # 这里假设图像已经被预处理为相同大小
            pixel_values = torch.stack([torch.tensor(list(image.getdata())).view(3, image.size[1], image.size[0]).float() for image in images])
        
        # 处理标签
        if label2id is not None:
            label_ids = torch.tensor([label2id[label] for label in labels])
        elif model and hasattr(model.config, 'label2id'):
            label_ids = torch.tensor([model.config.label2id[label] for label in labels])
        else:
            raise ValueError("label2id mapping is required but not provided")
        
        # 返回处理后的输入和标签
        return {
            'pixel_values': pixel_values,
            'labels': label_ids
        }
    
    def _process_text_features(self, features, model=None, dataset=None, label2id=None):
        """
        处理文本特征数据
        :param features: 文本特征列表
        :param model: 模型实例（可选）
        :param dataset: 数据集实例（可选）
        :param label2id: 标签到ID的映射（可选）
        :return: 处理后的数据字典
        """
        # 提取文本和标签
        texts = [item['text'] for item in features]
        labels = [item['label'] for item in features]
        
        # 处理文本
        if model and hasattr(model, 'tokenizer'):
            # 使用模型的tokenizer处理文本
            encoding = model.tokenizer(
                texts,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
        else:
            raise ValueError("Model with tokenizer is required for text processing")
        
        # 处理标签
        if label2id is not None:
            label_ids = torch.tensor([label2id[label] for label in labels])
        elif model and hasattr(model.config, 'label2id'):
            label_ids = torch.tensor([model.config.label2id[label] for label in labels])
        else:
            raise ValueError("label2id mapping is required but not provided")
        
        # 返回处理后的输入和标签
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }