from pathlib import Path
from os import path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_DIR, "data")

# files
SEOUL2JEJU_TSV = path.join(DATA_DIR, "seoul2jeju.tsv")
CONF_JSON = path.join(DATA_DIR, "conf.json")
