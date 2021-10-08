from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gib2kor
from dekorde.builders import build_I, build_M
from keras_preprocessing.text import Tokenizer
import torch


def main():
    conf = load_conf()
    # -- conf; hyper parameters --- #
    device = load_device()
    max_length = conf['max_length']
    embed_size = conf['embed_size']
    hidden_size = conf['hidden_size']
    heads = conf['heads']
    depth = conf['depth']

    # ---build the data --- #
    gib2kor = load_gib2kor()
    gibs = [gib for gib, _ in gib2kor]
    kors = [kor for _, kor in gib2kor]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    X = build_I(gibs, tokenizer, max_length, device)
    Y = build_I(kors, tokenizer, max_length, device)
    M = build_M(len(gib2kor), max_length, device)

    transformer = Transformer(embed_size, hidden_size, heads, depth)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=conf['lr'])

    for epoch in conf['epochs']:
        # compute the loss.
        loss = transformer.training_step(X, Y, M)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
