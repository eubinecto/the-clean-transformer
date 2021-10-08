
from dekorde.builders import build_M
from dekorde.loaders import load_device


def main():
    device = load_device()
    N = 10
    L = 30
    M = build_M(10, L, device)
    print(M[0])  # 배치 속 데이터 하나의 attention_mask.



if __name__ == '__main__':
    main()