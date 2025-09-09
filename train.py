#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用训练脚本
"""

import os
import sys
import argparse
from trainer.pipeline_factory import create_pipeline


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 创建pipeline
    pipeline = create_pipeline(args.config)
    
    # 开始训练
    pipeline.train()


if __name__ == '__main__':
    main()