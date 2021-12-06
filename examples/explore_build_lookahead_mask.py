
from dekorde.builders import build_lookahead_mask
from dekorde.loaders import load_device


def main():
    device = load_device()
    L = 30
    lookahead_mask = build_lookahead_mask(L, device)
    print(lookahead_mask)


if __name__ == '__main__':
    main()
