from pathlib import Path
from os import path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_DIR, "data")

# files
GIBBERISH2KOR_TSV = path.join(DATA_DIR, "gibberish2kor.tsv")
CONF_JSON = path.join(DATA_DIR, "conf.json")