import wandb
import argparse
from enkorde.builders import InferInputsBuilder
from enkorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer_scratch")
    parser.add_argument("--ver", type=str, default="overfit_small")
    parser.add_argument("--kor", type=str, default="그러나 이것은 또한 책상도 필요로 하지 않는다.")
    args = parser.parse_args()
    config = fetch_config()['train'][args.model][args.ver]
    config.update(vars(args))
    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        # fetch a pre-trained transformer with a pretrained tokenizer
        tokenizer = fetch_tokenizer(run, config['tokenizer'])
        transformer = fetch_transformer(run, config['model'], config['ver'])
        transformer.eval()
        X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=[config['kor']])
        src_ids = X[0, 0, 0].tolist()  # (1, 2, 2, L) -> (L) -> list
        pred_ids = transformer.predict(X).squeeze().tolist()  # (1, L) -> (L) -> list
        pred_ids = pred_ids[: pred_ids.index(tokenizer.eos_token_id)]  # noqa
        print(tokenizer.decode(ids=src_ids), "->", tokenizer.decode(ids=pred_ids))


if __name__ == '__main__':
    main()
