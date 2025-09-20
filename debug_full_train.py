import os
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForImageClassification, AutoConfig
from dataset import get_dataset
from utils.metric_utils import compute_metrics
import yaml

# 加载配置
config_path = 'examples/example_resnet_classifier.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

experiment_name = config['experiment_name']
model_args = config['model_args']
train_args = config['train_args']
dataset_args = config['dataset_args']

# 获取数据集
print("Loading dataset...")
dataset_module = get_dataset(dataset_args)
print("Dataset loaded")

print("Train dataset type:", type(dataset_module['train_dataset']))
if dataset_module['train_dataset']:
    print("Train dataset length:", len(dataset_module['train_dataset']))
    print("Train dataset features:", dataset_module['train_dataset'].features)
    
    # 查看原始数据的第一个样本
    print("\nOriginal first sample:")
    original_sample = dataset_module['train_dataset'][0]
    print("Original sample keys:", original_sample.keys())
    print("Original sample content:", {k: type(v) for k, v in original_sample.items()})

# 创建模型
print("\nCreating model...")
model_config = AutoConfig.for_model(**model_args)
model = AutoModelForImageClassification.from_config(model_config)
print("Model created")

# 定义训练和验证时的变换函数
def transform_train(examples):
    print("=== Transform Train ===")
    print("Input examples keys:", examples.keys() if hasattr(examples, 'keys') else 'Not a dict')
    if hasattr(examples, 'keys'):
        for k, v in examples.items():
            print(f"  {k}: {type(v)}")
    
    try:
        result = model.preprocess(examples, is_train=True)
        print("Output result keys:", result.keys() if hasattr(result, 'keys') else 'Not a dict')
        return result
    except Exception as e:
        print("Error in transform_train:", str(e))
        import traceback
        traceback.print_exc()
        raise

# 将变换函数应用到数据集
if dataset_module['train_dataset'] is not None:
    print("\nApplying transform to train dataset...")
    dataset_module['train_dataset'].set_transform(transform_train)

# 测试变换后的数据
if dataset_module['train_dataset']:
    print("\nTesting transformed data:")
    try:
        transformed_sample = dataset_module['train_dataset'][0]
        print("Transformed sample keys:", transformed_sample.keys())
        print("Transformed sample content:", {k: type(v) for k, v in transformed_sample.items()})
    except Exception as e:
        print("Error during transformation:", str(e))
        import traceback
        traceback.print_exc()