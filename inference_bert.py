import torch
from pretrained_model.bert_classifier import BertForTextClassification, BertClassifierConfig

def load_model(model_path):
    """加载训练好的BERT分类器模型"""
    # 从配置文件加载配置
    config = BertClassifierConfig.from_pretrained(model_path)
    
    # 创建模型
    model = BertForTextClassification.from_pretrained(model_path, config=config)
    
    # 设置为评估模式
    model.eval()
    
    return model

def predict_sentiment(model, text):
    """对文本进行情感分析"""
    # 预处理文本
    processed = model.preprocess({"text": text})
    
    # 添加批次维度
    input_ids = processed['input_ids'].unsqueeze(0) if processed['input_ids'].dim() == 1 else processed['input_ids']
    attention_mask = processed['attention_mask'].unsqueeze(0) if processed['attention_mask'].dim() == 1 else processed['attention_mask']
    
    # 进行推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # 获取预测标签
    id2label = model.config.id2label
    predicted_label = id2label[str(predictions.item())]
    
    # 获取预测概率
    probabilities = torch.softmax(logits, dim=-1)
    confidence = probabilities.max().item()
    
    return predicted_label, confidence

if __name__ == "__main__":
    # 加载模型
    model_path = "./experiments/0919_debug_train_bert_local/best"
    model = load_model(model_path)
    
    # 测试文本
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The worst film I've ever seen. Terrible acting and plot.",
        "Great story with excellent performances by all actors.",
        "Boring and predictable. Waste of time."
    ]
    
    # 对每个文本进行预测
    for text in test_texts:
        label, confidence = predict_sentiment(model, text)
        print(f"Text: {text}")
        print(f"Predicted sentiment: {label} (confidence: {confidence:.4f})")
        print("-" * 50)