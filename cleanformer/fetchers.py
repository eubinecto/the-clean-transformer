"""
what does it mean by "fetch"?: https://www.quora.com/What-does-fetch-means-in-computer-science
Korpora 에서도 fetch 라고 표현한다: https://github.com/ko-nlp/Korpora
"""
from os import path
import wandb
import yaml  # noqa
from typing import Tuple, List
from Korpora import KoreanParallelKOENNewsKorpus  # noqa
from tokenizers import Tokenizer  # noqa
from cleanformer.paths import CONFIG_YAML
from cleanformer.models.transformer import Transformer


def fetch_kor2eng(name: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    artifact = wandb.Api().artifact(f"cleanformer/{name}", type="dataset")
    train = [(row[0], row[1]) for row in artifact.get("train").data]
    val = [(row[0], row[1]) for row in artifact.get("val").data]
    test = [(row[0], row[1]) for row in artifact.get("test").data]
    return train, val, test


def fetch_tokenizer(name: str) -> Tokenizer:
    artifact = wandb.Api().artifact(f"cleanformer/{name}", type="other")
    artifact_path = artifact.download()
    json_path = path.join(artifact_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(json_path))
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata["pad"]
    tokenizer.pad_token_id = artifact.metadata["pad_id"]
    tokenizer.unk_token = artifact.metadata["unk"]
    tokenizer.unk_token_id = artifact.metadata["unk_id"]
    tokenizer.bos_token = artifact.metadata["bos"]
    tokenizer.bos_token_id = artifact.metadata["bos_id"]
    tokenizer.eos_token = artifact.metadata["eos"]
    tokenizer.eos_token_id = artifact.metadata["eos_id"]
    tokenizer.kor2eng = artifact.metadata['kor2eng']
    return tokenizer


def fetch_transformer(name: str) -> Transformer:
    artifact_path = wandb.Api().artifact(f"cleanformer/{name}", type="model").download()
    ckpt_path = path.join(artifact_path, "model.ckpt")
    transformer = Transformer.load_from_checkpoint(str(ckpt_path))
    return transformer


# --- fetchers for fetching local files --- #
def fetch_config() -> dict:
    """
    just load the config file from local
    """
    with open(str(CONFIG_YAML), "r") as fh:
        return yaml.safe_load(fh)
