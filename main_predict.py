import wandb
import argparse
from transformers import BertTokenizer
from dekorde.builders import InferInputsBuilder
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_config
from dekorde.paths import transformer_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jeju", type=str, default="딱 그 말을 허드라게")
    args = parser.parse_args()
    config = load_config()
    config.update(vars(args))
    with wandb.init(entity="eubinecto", project="dekorde") as run:
        artifact = run.use_artifact("transformer:latest")
        artifact.checkout()
    transformer_ckpt, tokenizer_dir = transformer_paths()
    transformer = Transformer.load_from_checkpoint(transformer_ckpt)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    transformer.eval()
    jejus = [config['jeju']]
    X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=jejus)
    tgt_ids = transformer.predict(X)
    for pred in tgt_ids.tolist():
        print(config['jeju'], "->", tokenizer.decode(pred))


if __name__ == '__main__':
    main()
