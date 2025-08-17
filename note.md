帮我构建一个项目，项目名称为SimpleTrainer。

项目介绍
SimpleTrainer是一个基于huggingface transformers的训练器，用于训练各种模型，包括图像分类、图像生成、文本生成等。
SimpleTrainer的目标是提供一个简单、灵活、可扩展的训练器，用于训练各种模型。

预期的项目结构
- dataset/ ：数据集的代码, 在这里实现torch dataset的实现 (待实现)
  对于image_dataset, 实现一个基础的SimpleImageDataset类, 继承torch dataset。返回pil格式的image和label，返回是dict格式。
- examples/ ：示例配置文件（暂时不需实现）
- network/ ：torch moduel类型的网络相关代码（暂时不需实现）
- pretrained_model/ ：huggingface格式的预训练模型类代码 (待实现)
    在pretrained_model/clip_vit_classifier.py 中实现huggingface格式的clip vit 分类器，有常规的config类和model类，model类继承自transformers PretrainedModel,并且拥有类方法from_clip_pretrained, 用于从openai/clip-vit-large-patch14模型初始化图像编码器和分类器头并加载权重。
    我的config类应该继承自transformers PretrainedConfig, 并实现from_clip_pretrained方法, 用于从openai/clip-vit-large-patch14模型初始化配置。
    我的config类应该额外持有label2id和id2label, 用于处理分类任务的标签。
    我的pretrained_model类应该持有processor, 用于处理输入的image和text, 并返回模型的输入。
    我的pretrained_model类应该持有loss_fct, 用于计算损失。
    我的model类应该继承自transformers PretrainedModel, 并实现forward方法, 用于前向传播。接受batch_image和label, 返回output。
    我的model类应该实现inference方法, 用于推理。接收pil格式的image, 并由id2label处理，返回分类结果字符串。
- utils/ ：工具类文件 (待实现)
- trainer/ ：训练器相关文件，包括各种训练pipeline实现 (待实现)
- requirements.txt ：项目依赖文件（已实现）
- main.py ：主程序入口，接受config.yaml文件作为参数 (待实现)

现在请先向我澄清你对项目的理解，以及实现计划，并向我确认是否需要修改。注意，得到我的同意后才能开始项目实现。


接下来请帮我实现trainer/ft_clip_vit_classifier.py和main.py。
ft_clip_vit_classifier.py实现clip vit分类器的训练pipeline类, 类名叫做CLIPViTFinetunePipeline, pipeline类的__init__方法接受config.yaml文件作为参数, 并在init方法中初始化dataset、model、processor、trainer。
在train方法中，调用trainer的train方法完成训练。train方法无参数，无返回值。
pipeline类中不要重新实现trainer，直接使用transformers Trainer类。
并在main.py中调用pipeline的train方法。
