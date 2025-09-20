# SimpleTrainer

SimpleTrainer 是一个通用的深度学习模型训练框架，支持多种模型架构和数据集类型。该项目基于 Hugging Face Transformers 库构建，提供了简洁的训练接口和灵活的配置选项。

## 功能特点

- **多模型支持**：支持 ResNet、ViT、Wav2Vec2、BERT 等多种模型架构
- **多任务训练**：支持图像分类、文本分类、音频分类等任务
- **灵活配置**：通过 YAML 配置文件管理训练参数
- **模块化设计**：清晰的代码结构，易于扩展和维护
- **多种数据源支持**：支持 Hugging Face Hub 数据集和本地 JSONL 格式数据集

## 项目结构

```
SimpleTrainer/
├── dataset/              # 数据集处理模块
│   ├── images/           # 图像数据集处理
│   ├── texts/            # 文本数据集处理
│   └── audios/           # 音频数据集处理
├── examples/             # 配置文件示例和示例数据
├── network/              # 网络模型定义（如ResNet等）
├── pretrained_model/     # 预训练模型定义和封装
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

### 使用 Hugging Face 数据集的配置示例

```yaml
### pipeline
stage: clst
experiment_name: example_bert_classifier

### model
model_args:
  model_type: bert_classifier
  id2label:
    0: "negative"
    1: "positive"
  label2id:
    "negative": 0
    "positive": 1
  model_name: bert-base-uncased
  num_classes: 2
  classifier_dropout: 0.1

### dataset
dataset_args:
  hf_dataset_name: "imdb"
  hf_dataset_config: null
  train_split: "train"
  eval_split: "test"

### train
train_args:
  logging_steps: 10
  save_steps: 500
  save_total_limit: 1
  num_train_epochs: 1
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  warmup_steps: 500
  weight_decay: 0.01
  eval_strategy: steps
  eval_steps: 500
  load_best_model_at_end: true
```

### 使用本地 JSONL 数据集的配置示例

```yaml
### pipeline
stage: clst
experiment_name: example_bert_classifier_local

### model
model_args:
  model_type: bert_classifier
  id2label:
    0: "negative"
    1: "positive"
  label2id:
    "negative": 0
    "positive": 1
  model_name: bert-base-uncased
  num_classes: 2
  classifier_dropout: 0.1

### dataset
dataset_args:
  dataset_type: text
  train_jsonline_path: examples/sample_text_data.jsonl
  val_jsonline_path: examples/sample_text_data.jsonl
  label2id:
    "negative": 0
    "positive": 1
  id2label:
    0: "negative"
    1: "positive"

### train
train_args:
  logging_steps: 10
  save_steps: 500
  save_total_limit: 1
  num_train_epochs: 5
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  warmup_steps: 500
  weight_decay: 0.01
  eval_strategy: steps
  eval_steps: 500
  load_best_model_at_end: true
```

## 支持的模型

- ResNet 图像分类器
- ViT 图像分类器
- Wav2Vec2 音频分类器
- BERT 文本分类器

## 数据集格式

SimpleTrainer 支持两种数据集格式：

### 1. Hugging Face Hub 数据集
直接使用 Hugging Face Hub 上的数据集，通过指定 `hf_dataset_name` 参数。

### 2. 本地 JSONL 格式数据集
使用本地 JSONL 格式的数据集，每行一个 JSON 对象，包含特征和标签。

#### 文本分类数据格式
```json
{"text": "This movie is absolutely fantastic!", "label": "positive"}
{"text": "The worst film I've ever seen.", "label": "negative"}
```

#### 图像分类数据格式
```json
{"image_path": "path/to/image1.jpg", "label": 0}
{"image_path": "path/to/image2.jpg", "label": 1}
```

## BERT 文本分类器使用说明

### 训练

要使用 BERT 文本分类器进行训练，可以按照以下步骤操作：

1. 准备数据集：创建一个 jsonl 格式的文件，每行包含一个文本样本和对应的标签，例如：
   ```json
   {"text": "This movie is absolutely fantastic!", "label": "positive"}
   {"text": "The worst film I've ever seen.", "label": "negative"}
   ```

2. 创建配置文件：参考 `examples/example_bert_classifier.yaml` 或 `examples/example_bert_classifier_local.yaml` 文件创建自己的配置文件。

3. 运行训练命令：
   ```bash
   python train.py --config your_config_file.yaml
   ```

## 许可证

本项目采用 MIT 许可证授权。详细信息请参见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。