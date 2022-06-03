"""
A script for building a tokenizer (fit your tokenizer to the data you have)
1. Byte Pair Encoding
2. Word Piece
# refer to the tutorial below.
https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
# for more details on normalizers, pre/post & tokenizers
https://huggingface.co/docs/tokenizers/python/latest/components.html
"""
import os
import wandb
import argparse
from itertools import chain
from tokenizers import pre_tokenizers, normalizers  # noqa
from tokenizers import Tokenizer  # noqa
from tokenizers.models import BPE, WordPiece  # noqa
from tokenizers.normalizers import Lowercase  # noqa
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation  # noqa
from tokenizers.trainers import BpeTrainer, WordPieceTrainer  # noqa
from cleanformer.paths import ROOT_DIR
from cleanformer.fetchers import fetch_config, fetch_kor2eng

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    config = fetch_config()["tokenizer"]
    config.update(vars(args))
    # --- prepare a tokenizer --- #
    special_tokens = [config["pad"], config["unk"], config["bos"], config["eos"]]
    tokenizer = Tokenizer(WordPiece(unk_token=config["unk"]))  # noqa
    trainer = WordPieceTrainer(vocab_size=config["vocab_size"], special_tokens=special_tokens)
    # --- pre & post processing --- #
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])  # noqa
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])  # noqa
    # --- prepare the data --- #
    kor2eng_train, kor2eng_val, kor2eng_test = fetch_kor2eng()
    # chaining two generators;  https://stackoverflow.com/a/3211047
    iterator = chain(
        (kor for kor, _ in kor2eng_train),
        (eng for _, eng in kor2eng_train),
        (kor for kor, _ in kor2eng_val),
        (eng for _, eng in kor2eng_val),
        (kor for kor, _ in kor2eng_test),
        (eng for _, eng in kor2eng_test),
    )
    # --- train the tokenizer --- #
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    # --- then save it --- #
    with wandb.init(project="cleanformer", config=config, tags=[__file__]) as run:
        # save to local, and then to wandb
        json_path = ROOT_DIR / "tokenizer.json"
        tokenizer.save(str(json_path), pretty=True)  # noqa
        artifact = wandb.Artifact(name="tokenizer", type="other", metadata=config)
        artifact.add_file(str(json_path))
        run.log_artifact(artifact)
    os.remove(str(json_path))  # make sure you delete it after you are done with uploading it


if __name__ == "__main__":
    main()
