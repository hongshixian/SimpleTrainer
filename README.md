# SimpleTrainer (简易丹炉)

SimpleTrainer是一个基于Hugging Face Transformers的训练器，用于训练各种模型，包括图像分类、图像生成、文本生成等。SimpleTrainer的目标是提供一个简单、灵活、可扩展的训练器，用于训练各种模型。

## 项目结构

- `dataset/`：数据集的代码，实现torch dataset
- `examples/`：示例配置文件
- `network/`：torch module类型的网络相关代码
- `pretrained_model/`：huggingface格式的预训练模型类代码
- `trainer/`：训练器相关文件，包括各种训练pipeline实现
- `utils/`：工具类文件
- `main.py`：主程序入口，接受config.yaml文件作为参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练CLIP ViT分类器

```bash
python main.py --config examples/train_full/clip_vit_classifier.yaml
```

### 训练BERT分类器

```bash
python main.py --config examples/train_full/bert_classifier.yaml
```

## 主要特性

- 简单易用：提供简洁的API，方便快速上手
- 灵活扩展：模块化设计，易于添加新的模型和训练器
- 多模型支持：支持多种类型的模型训练
- 基于Transformers：充分利用Hugging Face Transformers库的功能