import argparse
from cleanformer import preprocess as P  # noqa
from cleanformer.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="overfit_small")
    parser.add_argument("--src", type=str, default="카페인은 원래 커피에 들어있는 물질이다.")
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    # fetch a pre-trained transformer & and a pre-trained tokenizer
    transformer = fetch_transformer(config['ver'])
    tokenizer = fetch_tokenizer(config['tokenizer'])
    transformer.eval()  # otherwise, the result will be different on every run
    x2y = [(config['src'], "")]
    src = P.src(tokenizer, config['max_length'], x2y)
    tgt_r = P.tgt_r(tokenizer, config['max_length'], x2y)
    pred_ids = transformer.infer(src, tgt_r).squeeze().tolist()  # (1, L) -> (L) -> list
    pred_ids = pred_ids[:pred_ids.index(tokenizer.eos_token_id)]  # noqa, stop at the first eos token.
    src_ids = src[0, 1].tolist()  # (1, 2, L) -> (L) -> list
    print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))


if __name__ == '__main__':
    main()
