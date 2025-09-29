import os
import torch
from transformers import Trainer, TrainingArguments
from transformers import AutoModel, AutoConfig
from dataset import get_dataset
from utils.metric_utils import compute_metrics
import pretrained_model
from .wrapper import get_wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_clsft(
    experiment_name: str,
    model_args: dict,
    train_args: dict,
    dataset_args: dict,
    **kwargs
):
    #
    ckpt_save_dir = os.path.join(f"experiments/{experiment_name}", "ckpt")
    best_save_dir = os.path.join(f"experiments/{experiment_name}", "best")
    logs_dir = os.path.join("logs", experiment_name)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)
    os.makedirs(best_save_dir, exist_ok=True)
    # 
    dataset_module = get_dataset(dataset_args)  # dataset_module = {'train_dataset': datasets.Dataset, 'eval_dataset': datasets.Dataset}
    
    # 加载模型
    model = get_wrapper(model_args, device)
    
    # 定义训练和验证时的变换函数
    def transform_train(examples):
        return model.preprocess(examples, is_train=True)
    
    def transform_eval(examples):
        return model.preprocess(examples, is_train=False)
    
    # 将变换函数应用到数据集
    if dataset_module['train_dataset'] is not None:
        dataset_module['train_dataset'].set_transform(transform_train)
    if dataset_module['eval_dataset'] is not None:
        dataset_module['eval_dataset'].set_transform(transform_eval)
    
    #
    training_args = TrainingArguments(
        output_dir=ckpt_save_dir,
        logging_dir=logs_dir,
        report_to="tensorboard",
        remove_unused_columns=False,
        **train_args
    )
    #
    trainer = Trainer(
        model=model.model,
        args=training_args,
        compute_metrics=compute_metrics,
        **dataset_module
    )
    #
    trainer.train()
    #
    trainer.save_model(best_save_dir)
    #
    if "tokenizer" in model.__dict__:
        model.tokenizer.save_pretrained(best_save_dir)
    if "processor" in model.__dict__:
        model.processor.save_pretrained(best_save_dir)

    return