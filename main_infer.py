import argparse
from enkorde.builders import InferInputsBuilder
from enkorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    # you must provide this
    parser.add_argument("entity", type=str, help="a wandb entity to download artifacts from")
    parser.add_argument("--ver", type=str, default="overfit_small")
    parser.add_argument("--kor", type=str, default="결정적인 순간에 그들의 능력을 증가시켜 줄 그 무엇이 매우 중요합니다")
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    # fetch a pre-trained transformer & and a pre-trained tokenizer
    transformer = fetch_transformer(config['entity'], config['ver'])
    tokenizer = fetch_tokenizer(config['entity'], config['tokenizer'])
    transformer.eval()  # otherwise, the result will be different on every run
    X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=[config['kor']])
    src_ids = X[0, 0, 0].tolist()  # (1, 2, 2, L) -> (L) -> list
    pred_ids = transformer.predict(X).squeeze().tolist()  # (1, L) -> (L) -> list
    pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_token_id)]  # noqa
    print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))


if __name__ == '__main__':
    main()
