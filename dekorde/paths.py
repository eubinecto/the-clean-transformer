from pathlib import Path
import os

# directories
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
KORPORA_DIR = os.path.join(ROOT_DIR, "korpora")

# --- local files --- #
CONFIG_YAML = os.path.join(ROOT_DIR, "config.yaml")
