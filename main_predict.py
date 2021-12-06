from transformers import BertTokenizer
from dekorde.builders import InferInputsBuilder
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_config, load_seoul2jeju
from dekorde.paths import TRANSFORMER_CKPT, TOKENIZER_DIR


def main():
    config = load_config()
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
    transformer = Transformer.load_from_checkpoint(TRANSFORMER_CKPT)
    transformer.eval()
    seoul2jeju = load_seoul2jeju()[:10]
    seouls = [seoul for seoul, _ in seoul2jeju]
    jejus = [jeju for _, jeju in seoul2jeju]
    X = InferInputsBuilder(tokenizer, config['max_length'])(srcs=jejus)
    tgt_ids = transformer.predict(X)
    for pred, ans in zip(tgt_ids.tolist(), seouls):
        print(tokenizer.decode(pred), "|", ans)


if __name__ == '__main__':
    main()
