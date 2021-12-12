from cleanformer.tensors import subsequent_mask


def main():
    max_length = 10
    mask = subsequent_mask(max_length)
    print(mask)  # (L, L)
    print(mask)


if __name__ == '__main__':
    main()