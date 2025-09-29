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
from transformers import AutoModel, AutoConfig, Trainer, TrainingArguments
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


def main():
    args = get_args()
    
    # 加载模型
    model, config = load_trained_model(args.model_path, args.device)
    
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
            dataset_args['eval_split'] = dataset_split
        elif args.dataset_type == 'custom':
            dataset_args['custom_dataset_name'] = dataset_name
            dataset_args['eval_split'] = dataset_split
        else:
            # 对于jsonl类型，假设有一些默认的处理方式
            dataset_args['dataset_name'] = dataset_name
            dataset_args['eval_split'] = dataset_split
        
        # 获取评估数据集
        datasets = get_dataset(dataset_args)
        eval_dataset = datasets.get('eval_dataset')
        
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
        
        # 创建Trainer用于评估
        training_args = TrainingArguments(
            output_dir="./tmp_eval",  # 临时目录
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # 根据模型类型获取实际模型
        if hasattr(model, 'model'):
            eval_model = model.model
        else:
            eval_model = model
        
        # 创建评估器
        evaluator = Trainer(
            model=eval_model,
            args=training_args,
            compute_metrics=compute_metrics,
            eval_dataset=eval_dataset,
        )
        
        # 执行评估
        print("Starting evaluation with Trainer API...")
        eval_results = evaluator.evaluate()
        
        # 显示指标
        print(f"\nEvaluation Metrics for {dataset_name} ({dataset_split}):")
        for key, value in eval_results.items():
            if key not in ['epoch']:
                print(f"{key}: {value:.4f}")
        
        # 创建空的结果DataFrame（保持原有结构）
        results_data = {
            'predictions': [],
        }
        
        # 添加数据集信息到结果
        results_data['dataset_name'] = dataset_name
        results_data['dataset_split'] = dataset_split
        
        results_df = pd.DataFrame(results_data)
        
        # 添加数据集信息到指标
        eval_results['dataset_name'] = dataset_name
        eval_results['dataset_split'] = dataset_split
        
        # 收集结果
        all_results.append(results_df)
        all_metrics.append(eval_results)
    
    if not all_results:
        raise ValueError("No valid datasets found for evaluation")
    
    # 合并所有指标
    combined_metrics_df = pd.DataFrame(all_metrics)
    
    # 保存指标
    combined_metrics_df.to_csv(args.output_path, index=False)
    print(f"\nEvaluation metrics saved to {args.output_path}")


if __name__ == "__main__":
    main()