"""
what does it mean by "fetch"?: https://www.quora.com/What-does-fetch-means-in-computer-science
Korpora 에서도 fetch 라고 표현한다: https://github.com/ko-nlp/Korpora
"""
import yaml
from os import path
from typing import Tuple, List
from Korpora import KoreanParallelKOENNewsKorpus
from tokenizers import Tokenizer
from wandb.sdk.wandb_run import Run
from enkorde.paths import CONFIG_YAML, KORPORA_DIR, TOKENIZER_DIR, TRANSFORMER_DIR
from enkorde.models import Transformer


def fetch_kor2eng() -> Tuple[List[Tuple[str, str]],
                             List[Tuple[str, str]],
                             List[Tuple[str, str]]]:
    # download the data
    korpus = KoreanParallelKOENNewsKorpus(root_dir=KORPORA_DIR)
    kor2eng_train = list(zip(korpus.train.texts, korpus.train.pairs))
    kor2eng_val = list(zip(korpus.dev.texts, korpus.dev.pairs))
    kor2eng_test = list(zip(korpus.test.texts, korpus.test.pairs))
    return kor2eng_train, kor2eng_val, kor2eng_test


def fetch_tokenizer(run: Run, ver: str = "latest") -> Tokenizer:
    artifact = run.use_artifact(f"tokenizer:{ver}", type="other")
    # use checkout instead of download to save space and simplify directory structures
    # https://docs.wandb.ai/ref/python/artifact#checkout
    artifact_path = artifact.checkout(root=TOKENIZER_DIR)
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


def fetch_transformer(run: Run, ver: str = "latest") -> Transformer:
    artifact_path = run.use_artifact(f"transformer:{ver}", type="model")\
                       .checkout(root=TRANSFORMER_DIR)
    ckpt_path = path.join(artifact_path, "transformer.ckpt")
    transformer = Transformer.load_from_checkpoint(ckpt_path)
    return transformer


# --- fetchers for fetching local files --- #
def fetch_config() -> dict:
    """
    just load the config file from local
    """
    with open(CONFIG_YAML, 'r') as fh:
        return yaml.safe_load(fh)
