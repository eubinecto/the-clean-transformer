import wandb
import argparse
from dekorde.builders import InferInputsBuilder
from dekorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="overfit")
    parser.add_argument("--kor", type=str, default="안녕하세요")
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        # fetch a pre-trained transformer with a pretrained tokenizer
        transformer = fetch_transformer(run, config['ver'])
        tokenizer = fetch_tokenizer(run, config['tokenizer'])
        transformer.eval()
        kors = [config['kor']]
        X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=kors)
        srcs = X[:, 0, 0].squeeze().tolist()  # (1, 2, 2, L) -> (1, L) -> (L) -> list
        preds = transformer.predict(X).squeeze().tolist()  # (1, L) -> (L) -> list
        print([tokenizer.id_to_token(src) for src in srcs])
        print("->")
        print([tokenizer.id_to_token(pred) for pred in preds])


if __name__ == '__main__':
    main()
