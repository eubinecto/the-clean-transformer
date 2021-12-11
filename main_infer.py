import wandb
import argparse
from enkorde.builders import InferInputsBuilder
from enkorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer_torch")
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
        kors = [config['kor']]
        X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=kors)
        src_ids = X[:, 0, 0].squeeze().tolist()  # (1, 2, 2, L) -> (1, L) -> (L) -> list
        pred_ids = transformer.predict(X).squeeze().tolist()  # (1, L) -> (L) -> list
        print([tokenizer.id_to_token(src_id) for src_id in src_ids])
        print("->")
        print([tokenizer.id_to_token(pred_id) for pred_id in pred_ids])


if __name__ == '__main__':
    main()
