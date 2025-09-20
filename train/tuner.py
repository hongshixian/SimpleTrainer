import os
from typing import TYPE_CHECKING, Any, Optional
import argparse
import yaml
from .clst import run_clst


def read_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def _training_function(config: dict[str, Any]) -> None:
    stage = config.get("stage")

    if stage == "clst":
        run_clst(**config)
    else:
        raise ValueError(f"Unknown task: {stage}.")


def run_exp(args: argparse.Namespace) -> None:
    config = read_config(args)
    _training_function(config)
    