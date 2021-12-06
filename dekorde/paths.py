from pathlib import Path
from os import path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(ROOT_DIR, "data")
TOKENIZER_DIR = path.join(DATA_DIR, "tokenizer")
# files
SEOUL2JEJU_TSV = path.join(DATA_DIR, "seoul2jeju.tsv")
CONFIG_JSON = path.join(ROOT_DIR, "config.yaml")
TRANSFORMER_CKPT = path.join(DATA_DIR, "transformer.ckpt")
