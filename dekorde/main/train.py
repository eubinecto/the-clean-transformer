from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gib2kor
from dekorde.builders import build_X, build_Y, build_M
from keras_preprocessing.text import Tokenizer
import torch


def main():
    conf = load_conf()
    # -- conf; hyper parameters --- #
    device = load_device()
    max_length = conf['max_length']
    hidden_size = conf['hidden_size']
    heads = conf['heads']
    depth = conf['depth']
    lr = conf['lr']

    # ---build the data --- #
    gib2kor = load_gib2kor()
    gibs = [gib for gib, _ in gib2kor]
    kors = [kor for _, kor in gib2kor]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    vocab_size = len(tokenizer.word_index.keys())
    X = build_X(gibs, tokenizer, max_length, device)
    Y = build_Y(kors, tokenizer, max_length, device)
    M = build_M(max_length, device)

    # --- instantiate the model and the optimizer --- #
    transformer = Transformer(hidden_size, vocab_size, max_length, heads, depth, M)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=lr)

    # --- start training --- #
    for epoch in conf['epochs']:
        # compute the loss.
        loss = transformer.training_step(X, Y, M)
        loss.backward()  # backprop
        optimizer.step()  # gradient descent
        optimizer.zero_grad()  # prevent the gradients accumulating.


if __name__ == '__main__':
    main()
