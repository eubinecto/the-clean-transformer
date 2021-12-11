import wandb
import argparse
from enkorde.builders import InferInputsBuilder
from enkorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="overfit_small")
    parser.add_argument("--kor", type=str, default="양측은 또한 지구 온난화와 새 국제 형사 재판소를 포함한 광범위한 문제에 대해 견해 차이를 보여왔다.")
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
