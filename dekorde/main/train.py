from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_X, build_Y, build_M
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import torch


def main():
    conf = load_conf()
    device = load_device()
    gibberish2kor = load_gibberish2kor()

    gibs = [row[0] for row in gibberish2kor]
    kors = [row[1] for row in gibberish2kor]
    # integer encoding
    tokenizer = Tokenizer(char=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    X = build_X(gibberish2kor, tokenizer, conf['max_length'], device)  # (N, L)
    Y = build_Y(gibberish2kor, tokenizer, conf['max_length'], device)  # (N, L)
    M = build_M(gibberish2kor, device)  # (N, L, L)

    transformer = Transformer(conf['embed_size'], conf['hidden_size'], conf['heads'], conf['depth'])
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=conf['lr'])

    for epoch in conf['epochs']:
        loss = transformer.training_step(X, Y, M)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
