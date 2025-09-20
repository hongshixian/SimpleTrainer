# Wav2Vec2 音频分类器使用说明

## 简介

Wav2Vec2 音频分类器是基于 Facebook AI 的 Wav2Vec2 模型实现的音频分类器，可以用于关键词识别、语音命令识别等任务。

## 配置文件示例

```yaml
### pipeline
stage: clst
experiment_name: 0919_debug_train_wav2vec2

### model
model_args:
  model_type: wav2vec2_classifier
  id2label:
    0: "yes"
    1: "no"
    2: "up"
    3: "down"
    4: "left"
    5: "right"
    6: "on"
    7: "off"
    8: "stop"
    9: "go"
  label2id:
    "yes": 0
    "no": 1
    "up": 2
    "down": 3
    "left": 4
    "right": 5
    "on": 6
    "off": 7
    "stop": 8
    "go": 9
  model_name: facebook/wav2vec2-base
  num_labels: 10

### dataset
# 使用Hugging Face Hub数据集的配置示例
dataset_args:
  hf_dataset_name: "superb"
  hf_dataset_config: "ks"
  train_split: "train"
  eval_split: "validation"

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
  eval_steps: 20
  load_best_model_at_end: true
```

## 使用方法

1. 准备配置文件，参考 `examples/example_wav2vec2_classifier.yaml`
2. 运行训练命令：
   ```bash
   python train.py --config examples/example_wav2vec2_classifier.yaml
   ```

## 支持的数据集格式

### Hugging Face Hub 数据集

可以直接使用 Hugging Face Hub 上的音频数据集，如 SUPERB 数据集。

### JSONL 格式本地数据集

对于本地音频数据集，可以使用 JSONL 格式，每一行包含一个样本：

```json
{"audio_path": "/path/to/audio1.wav", "label": "yes"}
{"audio_path": "/path/to/audio2.wav", "label": "no"}
```

## 模型特性

- 支持冻结/解冻 Wav2Vec2 主干网络参数
- 自动处理音频预处理和特征提取
- 支持多种音频采样率的自动重采样