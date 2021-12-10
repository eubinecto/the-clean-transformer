"""
A script for building a tokenizer (fit your tokenizer to the data you have)
1. bpe
2. sentencepiece
3. wordpiece
# refer to the tutorial below.
https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
"""
import os
import wandb
import argparse
from itertools import chain
from tokenizers import pre_tokenizers, normalizers
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from dekorde.paths import ROOT_DIR
from dekorde.fetchers import fetch_config, fetch_kor2eng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="wp")
    args = parser.parse_args()
    config = fetch_config()["build"]
    config.update(vars(args))
    # --- prepare a tokenizer --- #
    if config['ver'] == "bpe":
        # tokenizer = Tokenizer(BPE(unk_token=config['unk']))
        tokenizer = Tokenizer(BPE(unk_token=config['unk']))
        trainer = BpeTrainer(vocab_size=config['vocab_size'])
    elif config['ver'] == "wp":
        tokenizer = Tokenizer(WordPiece(unk_token=config['unk']))  # noqa
        trainer = WordPieceTrainer(vocab_size=config['vocab_size'])
    else:
        raise ValueError(f"Invalid ver: {config['ver']}")
    # --- pre & post processing --- #
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])  # noqa
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])  # noqa
    # --- prepare the data --- #
    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        kor2eng_train, kor2eng_val, kor2eng_test = fetch_kor2eng()
        # chaining two generators;  https://stackoverflow.com/a/3211047
        iterator = chain((kor for kor, _ in kor2eng_train),
                         (eng for _, eng in kor2eng_train),
                         (kor for kor, _ in kor2eng_val),
                         (eng for _, eng in kor2eng_val),
                         (kor for kor, _ in kor2eng_test),
                         (eng for _, eng in kor2eng_test))
        # --- train the tokenizer --- #
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        # --- then save it --- #
        # save to local, and then to wandb
        json_path = os.path.join(ROOT_DIR, "tokenizer.json")
        tokenizer.save(json_path, pretty=True)  # noqa
        artifact = wandb.Artifact(name="tokenizer", type="other", metadata=config)
        artifact.add_file(json_path)
        run.log_artifact(artifact, aliases=["latest", config['ver']])
        os.remove(json_path)  # make sure you delete it after you are done with uploading it


if __name__ == '__main__':
    main()
