from dekorde.builders import build_M
from dekorde.loaders import load_device


def main():
    device = load_device()
    sents = [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
    ]
    L = 10
    M = build_M(sents, L, 8, device)
    print(M)


if __name__ == '__main__':
    main()
