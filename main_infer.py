import wandb
import argparse
from dekorde.builders import InferInputsBuilder
from dekorde.fetchers import fetch_config, fetch_tokenizer, fetch_transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="first")
    parser.add_argument("--jeju", type=str, default="딱 그 말을 허드라게")
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))
    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        # fetch a pre-trained transformer with a pretrained tokenizer
        transformer = fetch_transformer(run, config['ver'])
        tokenizer = fetch_tokenizer(run, config['tokenizer'])
        transformer.eval()
        jejus = [config['jeju']]
        X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=jejus)
        # (1, L) -> (L) -> python list
        preds = transformer.predict(X).squeeze().tolist()
        print(config['jeju'], "->", [tokenizer.id_to_token(pred) for pred in preds])


if __name__ == '__main__':
    main()
