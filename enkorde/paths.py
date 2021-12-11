from pathlib import Path
import os

# directories
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
WANDB_DIR = os.path.join(ROOT_DIR, "wandb")
KORPORA_DIR = os.path.join(ROOT_DIR, "korpora")

# --- local files --- #
CONFIG_YAML = os.path.join(ROOT_DIR, "config.yaml")


# --- artifacts --- #
TOKENIZER_DIR = os.path.join(ARTIFACTS_DIR, "tokenizer")
TRANSFORMER_TORCH_DIR = os.path.join(ARTIFACTS_DIR, "transformer_torch")
TRANSFORMER_SCRATCH_DIR = os.path.join(ARTIFACTS_DIR, "transformer_scratch")
