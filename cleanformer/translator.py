from typing import Tuple, List
from cleanformer.fetchers import fetch_config, fetch_transformer, fetch_tokenizer
from cleanformer import preprocess as P  # noqa


class Translator:
    """
    A Korean to English translator
    """

    def __init__(self):
        config = fetch_config()["recommended"]
        self.transformer = fetch_transformer(config["transformer"]).eval()
        self.tokenizer = fetch_tokenizer(config["tokenizer"])

    def __call__(self, sentences: List[str]) -> Tuple[List[str], List[str]]:
        x2y = [(sent, "") for sent in sentences]
        src = P.to_src(self.tokenizer, self.transformer.hparams["max_length"], x2y)
        tgt_r = P.to_tgt_r(self.tokenizer, self.transformer.hparams["max_length"], x2y)
        tgt_hat_ids = self.transformer.infer(src, tgt_r).tolist()  # (N, L) -> list
        src_ids = src[:, 0].tolist()  # (N, 2, L) -> (N, L) -> list
        inputs = self.tokenizer.decode_batch(src_ids, skip_special_tokens=True)
        predictions = self.tokenizer.decode_batch(tgt_hat_ids, skip_special_tokens=True)
        return inputs, predictions
