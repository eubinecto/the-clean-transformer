
from cleanformer.tensors import pos_encodings
import numpy as np


def main():
    N = 5
    max_length = 10
    hidden_size = 2
    encodings = pos_encodings(max_length, hidden_size)  # (L, H)
    # check if PE(pos + k) - PE(pos) stays constant
    print(np.linalg.norm(encodings[2] - encodings[0]))
    print(np.linalg.norm(encodings[3] - encodings[1]))
    print(np.linalg.norm(encodings[4] - encodings[2]))
    print(np.linalg.norm(encodings[5] - encodings[3]))
    # unlike repeat, expand does not copy tensors (saves GPU memory)
    # prefer view over reshape!
    # prefer expand over repeat!
    # (L, H) -> (1, L, H) -> (N, L, H)
    print(encodings.unsqueeze(0).expand(N, -1, -1).shape)  # (L, H) -> (N, L, H)


if __name__ == '__main__':
    main()
