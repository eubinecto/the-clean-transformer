import torch
import argparse
from cleanformer import preprocess as P  # noqa
from cleanformer.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="집이 정말 편안하네요")
    args = parser.parse_args()
    config = fetch_config()["transformer"]
    config.update(vars(args))
    # fetch a pre-trained transformer & and a pre-trained tokenizer
    transformer = fetch_transformer(config["best"])
    tokenizer = fetch_tokenizer(config["tokenizer"])
    with torch.no_grad():
        transformer.eval()  # otherwise, the result will be different on every run
        x2y = [(config["src"], "")]
        src = P.src(tokenizer, config["max_length"], x2y)
        tgt_r = P.tgt_r(tokenizer, config["max_length"], x2y)
        pred_ids = transformer.infer(src, tgt_r).squeeze().tolist()  # (1, L) -> (L) -> list
        pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_token_id)]  # noqa
        src_ids = src[0, 0].tolist()  # (1, 2, L) -> (L) -> list
    print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))


if __name__ == "__main__":
    main()
