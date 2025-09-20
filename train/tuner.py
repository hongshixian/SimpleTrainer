import os
from typing import TYPE_CHECKING, Any, Optional
import argparse
import yaml
from .ct import run_ct

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="examples/example_resnet_classifier.yaml", help="Path to the yaml config file.")

def get_args() -> argparse.Namespace:
    args = parser.parse_args()
    return args


def read_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _training_function(config: dict[str, Any]) -> None:
    stage = config.get("stage")

    if stage == "ct":
        run_ct(**config)
    else:
        raise ValueError(f"Unknown task: {stage}.")


def run_exp() -> None:
    args = get_args()
    config = read_config(args)
    _training_function(config)
    