"""
https://stackoverflow.com/a/47867513
"""

import torch

# a vector
x = torch.Tensor([1, 2, 2, 3, 3, 3, 4])

print(x == 2)
# the first index of 2
print((x == 2).nonzero()[0])


# what about a matrix?
print("---")
x = torch.Tensor([[1, 2, 2, 3, 3, 3, 4], [1, 3, 3, 2, 2, 2, 4]])
# the first index of 2
print((x == 2))
print((x == 2).nonzero())
