from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_X, build_Y, build_M
import torch


def main():
    conf = load_conf()
    device = load_device()
    gibberish2kor = load_gibberish2kor()
    X = build_X(gibberish2kor, device)
    Y = build_Y(gibberish2kor, device)
    M = build_M(gibberish2kor, device)

    transformer = Transformer(conf['embed_size'], conf['hidden_size'], conf['heads'], conf['depth'])
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=conf['lr'])

    for epoch in conf['epochs']:
        loss = transformer.training_step(X, Y, M)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == '__main__':
    main()
