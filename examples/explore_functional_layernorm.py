
import torch


def main():
    N = 10
    L = 30
    H = 768
    # https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    inputs_hidden = torch.rand((N, L, H))
    inputs_hidden = torch.layer_norm(inputs_hidden, (H,))
    print(inputs_hidden.shape)


if __name__ == '__main__':
    main()
