import yaml
from trainer.ft_clip_vit_classifier import CLIPViTFinetunePipeline
from trainer.ft_bert_classifier import BertFinetunePipeline
from trainer.ft_resnet_classifier import ResNetFinetunePipeline


def create_pipeline(config_path):
    """
    根据配置文件创建pipeline实例
    :param config_path: 配置文件路径
    :return: 相应的pipeline实例
    """
    # 读取配置文件以确定使用的pipeline
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取pipeline类型
    pipeline_type = config.get('pipeline_type', 'clip_vit')
    
    # 根据pipeline类型创建相应的pipeline实例
    if pipeline_type == 'clip_vit':
        return CLIPViTFinetunePipeline(config_path)
    elif pipeline_type == 'bert':
        return BertFinetunePipeline(config_path)
    elif pipeline_type == 'resnet':
        return ResNetFinetunePipeline(config_path)
    else:
        raise ValueError(f"Unsupported pipeline type: {pipeline_type}")