from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_I, build_M
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import torch


def main():
    conf = load_conf()

    # ========== loading conf ========== #
    device = load_device()
    hidden_size = conf.hidden_size
    embed_size = conf.embed_size
    heads = conf.heads
    depth = conf.depth
    epochs = conf.epochs
    max_length = conf.max_length
    lr = conf.lr

    # ========== loading data ========== #
    gibberish2kor = load_gibberish2kor()

    gibs = [row[0] for row in gibberish2kor]
    kors = [row[1] for row in gibberish2kor]

    # ========== setting tokenizer ========== #
    tokenizer = Tokenizer(char=True)
    tokenizer.fit_on_texts(texts=gibs + kors)

    # ========== converting raw text to tensor ========== #
    X = build_I(gibs, tokenizer, max_length, device)  # (N, L)
    Y = build_I(kors, tokenizer, max_length, device)  # (N, L)

    M = build_M()  # (N, L, L)

    # ========== loading model & opts ========== #
    transformer = Transformer(
        embed_size=embed_size,
        vocab_size=...,
        hidden_size=hidden_size,
        max_length=max_length,
        heads=heads,
        depth=depth
    )
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=conf['lr'])

    for _ in epochs:
        loss = transformer.training_step(X, Y, M)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
