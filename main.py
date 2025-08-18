import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer.pipeline_factory import create_pipeline

def main():
    """
    主程序入口
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SimpleTrainer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")
    
    # 使用create_pipeline函数创建pipeline实例
    pipeline = create_pipeline(args.config)
    
    # 调用pipeline的train方法
    pipeline.train()


if __name__ == "__main__":
    main()