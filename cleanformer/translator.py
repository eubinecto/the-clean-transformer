from typing import Tuple
from cleanformer.fetchers import fetch_config, fetch_transformer, fetch_tokenizer
from cleanformer import preprocess as P  # noqa


class Translator:
    def __init__(self):
        config = fetch_config()["recommended"]
        self.transformer = fetch_transformer(config["transformer"]).eval()
        self.tokenizer = fetch_tokenizer(config["tokenizer"])

    def __call__(self, kor: str) -> Tuple[str, str]:
        x2y = [(kor, "")]
        src = P.to_src(self.tokenizer, self.transformer.hparams["max_length"], x2y)
        tgt_r = P.to_tgt_r(self.tokenizer, self.transformer.hparams["max_length"], x2y)
        pred_ids = self.transformer.infer(src, tgt_r).squeeze().tolist()  # (1, L) -> (L) -> list
        pred_ids = pred_ids[: pred_ids.index(self.tokenizer.eos_token_id)]  # noqa
        src_ids = src[0, 0].tolist()  # (1, 2, L) -> (L) -> list
        return self.tokenizer.decode(src_ids), self.tokenizer.decode(pred_ids)
