import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    """
    计算评估指标
    :param eval_pred: 评估预测结果
    :return: 指标字典
    """
    predictions, labels = eval_pred
    
    # 如果predictions是logits，需要转换为预测类别
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    
    # 转换为numpy数组
    predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class MetricTracker:
    """
    指标跟踪器
    """
    def __init__(self):
        """
        初始化指标跟踪器
        """
        self.metrics = {}
        
    def update(self, metric_name, value):
        """
        更新指标
        :param metric_name: 指标名称
        :param value: 指标值
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def get_average(self, metric_name):
        """
        获取指标平均值
        :param metric_name: 指标名称
        :return: 平均值
        """
        if metric_name in self.metrics:
            return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
        return 0.0
        
    def reset(self):
        """
        重置指标
        """
        self.metrics = {}