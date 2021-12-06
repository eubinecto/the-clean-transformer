from typing import List, Tuple
from dekorde.paths import CONFIG_JSON,  SEOUL2JEJU_TSV
import yaml


def load_config() -> dict:
    with open(CONFIG_JSON, 'r') as fh:
        return yaml.safe_load(fh)

