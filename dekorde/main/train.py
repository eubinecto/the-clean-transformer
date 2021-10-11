from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_I, build_M
from keras_preprocessing.text import Tokenizer
import torch


def main():
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

    M = build_M(kors, head_size, max_length, device)  # (N, L, L)

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

    print('START')
    for epoch in range(epochs):
        loss = transformer.training_step(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch:{epoch}, loss:{loss}")


if __name__ == '__main__':
    main()
