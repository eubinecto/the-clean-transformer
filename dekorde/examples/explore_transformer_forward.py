import torch
from keras_preprocessing.text import Tokenizer

from dekorde.builders import build_I, build_M
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_device, load_conf, load_gibberish2kor


def explore_transformer_forward():
    conf = load_conf()

    # ========== loading conf ========== #
    device = load_device()
    d_model = conf.embed_size
    head_size = conf.heads
    depth = conf.depth
    epochs = conf.epochs
    max_length = conf.max_length
    lr = conf.lr

    # ========== loading data ========== #
    gibberish2kor = load_gibberish2kor()

    gibs = [row[0] for row in gibberish2kor]
    kors = [row[1] for row in gibberish2kor]

    # ========== setting tokenizer ========== #
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    vocab_size = len(tokenizer.word_index.keys())

    # ========== converting raw text to tensor ========== #
    X = build_I(gibs, tokenizer, max_length, device)  # (N, L)
    Y = build_I(kors, tokenizer, max_length, device)  # (N, L)

    M = build_M(kors, max_length, head_size, device)  # (N, L, L)

    # ========== loading model & opts ========== #
    transformer = Transformer(
        d_model=d_model,
        vocab_size=vocab_size,
        max_length=max_length,
        head_size=head_size,
        depth=depth,
        mask=M
    )
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=lr)

    res = transformer.forward(X, Y)

    print(res)

    logits = torch.einsum('nlh,vh->nlv', res, transformer.token_embeddings.weight)

    print(logits)

    probs = torch.softmax(logits, dim=0)

    print(probs)

    preds = torch.argmax(probs, dim=-1)

    print(preds)


if __name__ == '__main__':
    explore_transformer_forward()
