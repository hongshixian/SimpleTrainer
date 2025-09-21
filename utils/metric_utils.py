import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def classification_metrics(logits, labels):
    """
    计算多分类指标
    :param logits: 模型输出logits
    :param labels: 真实标签
    :return: 多分类指标字典
    """
    # 转换为numpy数组
    logits = logits.numpy() if isinstance(logits, torch.Tensor) else logits
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    # 获取预测类别
    predictions = np.argmax(logits, axis=-1)
    
    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def binary_metrics(positive_scores, binary_labels):
    """
    计算二分类指标
    :param positive_scores: 正样本分数
    :param binary_labels: 二分类标签
    :return: 二分类指标字典
    """
    # 转换为numpy数组
    positive_scores = positive_scores.numpy() if isinstance(positive_scores, torch.Tensor) else positive_scores
    binary_labels = binary_labels.numpy() if isinstance(binary_labels, torch.Tensor) else binary_labels
    
    # 计算ACC
    predictions = (positive_scores > 0.5).astype(int)
    accuracy = accuracy_score(binary_labels, predictions)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(binary_labels, positive_scores)
    
    # 计算EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    
    # 获取最接近0.5阈值的点
    idx = np.argmin(np.abs(thresholds - 0.5))
    
    return {
        'binary_accuracy': accuracy,
        'eer': eer,
        'tpr': tpr[idx],
        'fpr': fpr[idx],
        'threshold': thresholds[idx],
        'eer_threshold': eer_threshold
    }


def compute_metrics(eval_pred):
    """
    计算评估指标
    :param eval_pred: 评估预测结果，包含logits和labels
    :return: 合并后的指标字典
    """
    logits, labels = eval_pred
    
    # 转换为torch tensor如果需要的话
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    
    # 多分类指标计算
    cls_metrics = classification_metrics(logits, labels)
    
    # 二分类指标计算
    # 将logits的0位置作为negative，1和之后位置求和作为positive
    negative_score = logits[:, 0]
    positive_score = torch.sum(logits[:, 1:], dim=1)
    
    # 构造二分类标签：0为negative类，1为positive类
    binary_labels = (labels != 0).int()
    binary_scores = positive_score  # 使用正类的分数作为二分类分数
    
    bin_metrics = binary_metrics(binary_scores, binary_labels)
    
    # 合并两个指标字典
    all_metrics = {**cls_metrics, **bin_metrics}
    
    return all_metrics
