"""Helper utility to load YAML training recipes and merge with CLI arguments."""

import argparse
import os
import sys
from typing import Optional
import yaml


def apply_recipe(args: argparse.Namespace, default_recipe: Optional[str] = None) -> argparse.Namespace:
    """Load a YAML recipe file and populate missing/default args without overriding explicit CLI flags."""
    recipe_path = getattr(args, "recipe", None) or default_recipe
    if recipe_path and os.path.exists(recipe_path):
        with open(recipe_path, "r", encoding="utf-8") as f:
            recipe_data = yaml.safe_load(f) or {}

        for k, v in recipe_data.items():
            if hasattr(args, k):
                cli_flags = [f"--{k}", f"--{k.replace('_', '-')}"]
                was_passed = any(arg.split("=")[0] in cli_flags for arg in sys.argv[1:])
                if not was_passed:
                    setattr(args, k, v)
    return args
