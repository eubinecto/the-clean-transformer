from pathlib import Path
import os
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# files
CONFIG_JSON = os.path.join(ROOT_DIR, "config.yaml")


def transformer_paths() -> Tuple[str, str]:
    transformer_dir = os.path.join(ARTIFACTS_DIR, "transformer")
    transformer_ckpt = os.path.join(transformer_dir, "transformer.ckpt")
    tokenizer_dir = os.path.join(transformer_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    return transformer_ckpt, tokenizer_dir

# lstm paths

# rnn paths