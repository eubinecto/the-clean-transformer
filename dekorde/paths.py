from pathlib import Path
from os import path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(ROOT_DIR, "data")
# files
SEOUL2JEJU_TSV = path.join(DATA_DIR, "seoul2jeju.tsv")
CONFIG_JSON = path.join(ROOT_DIR, "config.yaml")
TRANSFORMER_BIN = path.join(DATA_DIR, "transformer.bin")
