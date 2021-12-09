"""
what does it mean by "fetch"?: https://www.quora.com/What-does-fetch-means-in-computer-science
Korpora 에서도 fetch 라고 표현한다: https://github.com/ko-nlp/Korpora
"""
import yaml
from os import path
from typing import Tuple, List
from tokenizers import Tokenizer
from wandb.sdk.wandb_run import Run
from dekorde.paths import CONFIG_YAML
from dekorde.models import Transformer
from pytorch_lightning import LightningModule


# --- fetchers for fetching artifacts --- #
def fetch_jeju2seoul(run: Run, ver: str = "latest") -> List[Tuple[str, str]]:
    table = run.use_artifact(f"seoul2jeju:{ver}") \
               .get("seoul2jeju")
    seoul2jeju = [(row[1], row[0]) for row in table.data]
    return seoul2jeju


def fetch_tokenizer(run: Run, ver: str = "latest") -> Tokenizer:
    artifact = run.use_artifact(f"tokenizer:{ver}", type="other")
    artifact_path = artifact.checkout()
    json_path = path.join(artifact_path, "tokenizer.json")
    tokenizer = Tokenizer.from_file(json_path)
    # just manually register the special tokens
    tokenizer.pad_token = artifact.metadata['pad']
    tokenizer.pad_token_id = artifact.metadata['pad_id']
    tokenizer.unk_token = artifact.metadata['unk']
    tokenizer.unk_token_id = artifact.metadata['unk_id']
    tokenizer.bos_token = artifact.metadata['bos']
    tokenizer.bos_token_id = artifact.metadata['bos_id']
    tokenizer.eos_token = artifact.metadata['eos']
    tokenizer.eos_token_id = artifact.metadata['eos_id']
    return tokenizer


def fetch_transformer(run: Run, ver: str = "latest") -> LightningModule:
    artifact_path = run.use_artifact(f"transformer:{ver}", type="model")\
                       .checkout()
    ckpt_path = path.join(artifact_path, "transformer.ckpt")
    transformer = Transformer.load_from_checkpoint(ckpt_path)
    return transformer


def fetch_lstm(run: Run, ver: str = "latest") -> LightningModule:  # noqa
    """
    to be added later
    :param run:
    :param ver:
    :return:
    """
    raise NotImplementedError


def fetch_rnn(run: Run, ver: str = "latest") -> LightningModule:  # noqa
    """
    to be added later
    :param run:
    :param ver:
    :return:
    """
    raise NotImplementedError


# --- fetchers for fetching local files --- #
def fetch_config() -> dict:
    """
    just load the config file from local
    """
    with open(CONFIG_YAML, 'r') as fh:
        return yaml.safe_load(fh)
