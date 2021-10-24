from typing import List, Tuple
from dekorde.paths import CONF_JSON,  SEOUL2JEJU_TSV
import torch
import json
import csv


def load_seoul2jeju() -> List[Tuple[str, str]]:
    with open(SEOUL2JEJU_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        next(tsv_reader)  # skip the header (seoul, jeju)
        return [
            (row[0], row[1])
            for row in tsv_reader
        ]


def load_conf() -> dict:
    with open(CONF_JSON, 'r') as fh:
        conf = json.loads(fh.read())
    return conf


def load_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
