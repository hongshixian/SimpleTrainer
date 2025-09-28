#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用训练脚本
"""
import os
import argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from train.tuner import run_exp

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, 
        default="examples/example_clsft_vit_classifier.yaml", 
        help="Path to the yaml config file.")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run_exp(args)


if __name__ == "__main__":
    main()