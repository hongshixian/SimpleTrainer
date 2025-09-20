#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用训练脚本
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from train.tuner import run_exp

def main():
    run_exp()


if __name__ == "__main__":
    main()