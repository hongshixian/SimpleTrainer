# SimpleTrainer

SimpleTrainer 是一个通用的深度学习模型训练框架，支持多种模型架构和数据集类型。该项目基于 Hugging Face Transformers 库构建，提供了简洁的训练接口和灵活的配置选项。

## 功能特点

- **多模型支持**：支持 ResNet、CLIP-ViT 等多种模型架构
- **多任务训练**：支持图像分类等任务
- **灵活配置**：通过 YAML 配置文件管理训练参数
- **模块化设计**：清晰的代码结构，易于扩展和维护
- **对比训练**：支持对比学习训练策略

## 项目结构

```
SimpleTrainer/
├── dataset/              # 数据集处理模块
│   ├── images/           # 图像数据集处理
│   ├── texts/            # 文本数据集处理
│   └── audios/           # 音频数据集处理
├── examples/             # 配置文件示例
├── network/              # 网络模型定义
├── pretrained_model/     # 预训练模型定义
├── train/                # 训练流程实现
│   └── clst/             # 分类器训练流程
├── utils/                # 工具函数
├── train.py              # 训练入口文件
└── requirements.txt      # 项目依赖
```

## 安装依赖

在开始使用之前，请确保安装了所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

1. **准备配置文件**：
   复制并修改 `examples/` 目录下的示例配置文件，或创建新的 YAML 配置文件。

2. **启动训练**：
   ```bash
   python train.py --config path/to/your/config.yaml
   ```

## 可视化训练进度

在训练过程中，SimpleTrainer 会自动记录训练指标到 `logs` 目录中。您可以使用 TensorBoard 来可视化这些指标：

```bash
tensorboard --logdir logs
```

然后在浏览器中打开 http://localhost:6006 来查看训练进度和指标。

## 配置文件说明

配置文件使用 YAML 格式，主要包含以下几个部分：

- `stage`: 训练阶段（目前支持 `clst` 分类器训练）
- `experiment_name`: 实验名称
- `model_args`: 模型参数配置
- `dataset_args`: 数据集配置
- `train_args`: 训练参数配置

示例配置文件：
```yaml
stage: clst
experiment_name: example_experiment

model_args:
  model_type: resnet_classifier
  id2label:
    0: "class_0"
    1: "class_1"

dataset_args:
  hf_dataset_name: "path/to/dataset"
  train_split: "train"
  eval_split: "validation"

train_args:
  num_train_epochs: 5
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
```

## 支持的模型

- ResNet 分类器
- CLIP-ViT 分类器

## 许可证

本项目采用 [许可证信息] 授权。详细信息请参见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。