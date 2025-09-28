#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估脚本
用于加载已训练好的模型并在数据集上进行测试评估
"""

import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import argparse
import pandas as pd
import torch
from transformers import AutoModel, AutoConfig
from dataset.dataset_loader import get_dataset
from utils.metric_utils import compute_metrics
import pretrained_model
from train.clsft.wrapper import get_wrapper


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument(
        "--model_path", "-m", type=str, default="experiments/0919_debug_train_wav2vec2/best",
        help="Path to the trained model directory")
    parser.add_argument(
        "--dataset_names", "-d", type=str, nargs='+', default=["ASVSpoof2019LA"],
        help="List of dataset names to evaluate on")
    parser.add_argument(
        "--dataset_splits", "-s", type=str, nargs='+',  default=["eval"],
        help="List of dataset splits corresponding to dataset names")
    parser.add_argument(
        "--dataset_type", "-t", type=str, default="custom",
        help="Type of dataset (hf for HuggingFace, custom, or jsonl)")
    parser.add_argument(
        "--output_path", "-o", type=str, default="evaluation_results.csv",
        help="Path to save evaluation results")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32,
        help="Batch size for evaluation")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (cuda/cpu)")
    
    args = parser.parse_args()
    return args


def load_trained_model(model_path, device):
    """
    加载已训练好的模型
    :param model_path: 模型路径
    :param device: 设备
    :return: 加载的模型和配置
    """
    print(f"Loading model from {model_path}")
    
    # 首先加载模型配置
    config = AutoConfig.from_pretrained(model_path)
    
    # 如果配置中有finetuning_task字段，则是HF模型
    if hasattr(config, 'finetuning_task') and config.finetuning_task:
        print("Loading as HF model...")

        # 使用包装器加载HF模型
        model_args = {
            "pretrained_model_name_or_path": model_path,
            "finetuning_task": config.finetuning_task
        }
        
        model_wrapper = get_wrapper(model_args, device)
        print(f"Loaded HF model with wrapper for task: {config.finetuning_task}")
        return model_wrapper, config

    # 否则是自定义模型
    else:
        print("Loading as custom model...")
        # 回退到使用AutoModel
        model = AutoModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
        print(f"Loaded model with AutoModel on {device}")
        return model, config


def evaluate_model(model, config, dataset, device, batch_size):
    """
    在数据集上评估模型
    :param model: 模型
    :param config: 配置
    :param dataset: 数据集
    :param device: 设备
    :param batch_size: 批次大小
    :return: 预测结果和指标
    """
    print("Starting evaluation...")
    
    # 存储所有预测结果和真实标签
    all_predictions = []
    all_labels = []
    all_logits = []
    
    # 获取数据集类型
    dataset_type = getattr(config, 'dataset_type', 'image')  # 默认为image
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=getattr(dataset, 'collate_fn', None)
    )
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
                
            # 检查batch是否已经是处理好的数据
            # 如果数据集已经应用了preprocess方法，batch应该已经是模型可以直接使用的格式
            # 将数据移动到设备
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # 前向传播
            try:
                outputs = model(**batch)
            except Exception as e:
                # 如果直接调用失败，尝试只传入必要的参数
                if 'input_ids' in batch and 'attention_mask' in batch:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                elif 'pixel_values' in batch:
                    outputs = model(pixel_values=batch['pixel_values'])
                elif 'input_values' in batch:
                    outputs = model(input_values=batch['input_values'])
                else:
                    raise e
                    
            # 检查outputs是否有logits属性
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # 如果没有logits属性，假设outputs本身就是logits
                logits = outputs
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            
            # 检查标签在哪个键中
            label_key = None
            for key in ['labels', 'label']:
                if key in batch:
                    label_key = key
                    break
            
            if label_key:
                all_labels.extend(batch[label_key].cpu().numpy())
            else:
                print("Warning: No labels found in batch")
    
    # 计算指标
    print("Computing metrics...")
    if all_labels:
        eval_pred = (all_logits, all_labels)
        metrics = compute_metrics(eval_pred)
    else:
        # 如果没有标签，只返回预测结果
        metrics = {}
        print("Warning: No labels available for metrics computation")
    
    # 创建结果DataFrame
    results_data = {
        'predictions': all_predictions,
    }
    
    if all_labels:
        results_data['labels'] = all_labels
    
    results_df = pd.DataFrame(results_data)
    
    return results_df, metrics


def main():
    args = get_args()
    
    # 加载模型
    model, config = load_trained_model(args.model_path, args.device)
    
    # 检查模型是否有preprocess方法
    has_preprocess = hasattr(model, 'preprocess')
    print(f"Model has preprocess method: {has_preprocess}")
    if not has_preprocess:
        raise NotImplementedError("Warning: Model does not have a preprocess method. Evaluation may not work as expected.")
    
    # 检查数据集名称和分割数量是否匹配
    if len(args.dataset_names) != len(args.dataset_splits):
        raise ValueError("Number of dataset names must match number of dataset splits")
    
    # 为每个数据集执行评估
    all_results = []
    all_metrics = []
    
    for dataset_name, dataset_split in zip(args.dataset_names, args.dataset_splits):
        print(f"\nEvaluating on dataset: {dataset_name}, split: {dataset_split}")
        
        # 构建数据集参数
        dataset_args = {
            'dataset_type': args.dataset_type
        }
        
        # 根据数据集类型设置相应的参数
        if args.dataset_type == 'hf':
            dataset_args['hf_dataset_name'] = dataset_name
            dataset_args['train_split'] = dataset_split
        elif args.dataset_type == 'custom':
            dataset_args['custom_dataset_name'] = dataset_name
            dataset_args['train_split'] = dataset_split
        else:
            # 对于jsonl类型，假设有一些默认的处理方式
            dataset_args['dataset_name'] = dataset_name
            dataset_args['train_split'] = dataset_split
        
        # 获取评估数据集
        datasets = get_dataset(dataset_args)
        eval_dataset = datasets.get('train_dataset')
        
        if eval_dataset is None:
            print(f"Warning: No evaluation dataset found for {dataset_name} ({dataset_split})")
            raise ValueError(f"Evaluation dataset not found for {dataset_name} ({dataset_split})")
        
        print(f"Evaluation dataset loaded with {len(eval_dataset)} samples")
        
        # 如果模型有preprocess方法，将其设置到数据集中
        # 定义验证时的变换函数
        def transform_eval(examples):
            return model.preprocess(examples, is_train=False)
        
        # 将变换函数应用到数据集
        if hasattr(eval_dataset, 'set_transform'):
            eval_dataset.set_transform(transform_eval)
            print("Applied preprocess method to dataset using set_transform")
        else:
            print("Dataset does not support set_transform method")
            raise NotImplementedError("Warning: Dataset does not support set_transform method. Evaluation may not work as expected.")
        
        # 执行评估
        # 如果使用包装器模型，需要传递model.model而不是model
        eval_model = model.model if hasattr(model, 'model') else model
        results_df, metrics = evaluate_model(
            eval_model, config, eval_dataset, args.device, args.batch_size
        )
        
        # 添加数据集信息到结果
        results_df['dataset_name'] = dataset_name
        results_df['dataset_split'] = dataset_split
        
        # 添加数据集信息到指标
        metrics['dataset_name'] = dataset_name
        metrics['dataset_split'] = dataset_split
        
        # 收集结果
        all_results.append(results_df)
        all_metrics.append(metrics)
        
        # 显示指标
        print(f"\nEvaluation Metrics for {dataset_name} ({dataset_split}):")
        for key, value in metrics.items():
            if key not in ['dataset_name', 'dataset_split']:
                print(f"{key}: {value:.4f}")
    
    if not all_results:
        raise ValueError("No valid datasets found for evaluation")
    
    # 合并所有结果
    combined_results_df = pd.concat(all_results, ignore_index=True)
    combined_metrics_df = pd.DataFrame(all_metrics)
    
    # 保存结果
    combined_results_df.to_csv(args.output_path, index=False)
    print(f"\nDetailed results saved to {args.output_path}")
    
    # 保存指标
    metrics_path = args.output_path.replace('.csv', '_metrics.csv')
    combined_metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()