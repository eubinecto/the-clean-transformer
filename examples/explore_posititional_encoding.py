
from tensors import pos_encodings
import numpy as np


def main():
    max_length = 10
    hidden_size = 2
    encodings = pos_encodings(max_length, hidden_size)  # (L, H)
    # check if PE(pos + k) - PE(pos) stays constant
    print(np.linalg.norm(encodings[2] - encodings[0]))
    print(np.linalg.norm(encodings[3] - encodings[1]))
    print(np.linalg.norm(encodings[4] - encodings[2]))
    print(np.linalg.norm(encodings[5] - encodings[3]))


if __name__ == '__main__':
    main()
