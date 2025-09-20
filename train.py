#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用训练脚本
"""
from train.tuner import run_exp
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    run_exp()


if __name__ == "__main__":
    main()