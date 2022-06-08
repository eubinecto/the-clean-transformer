"""
https://stackoverflow.com/a/47867513
"""
import torch

# a vector
eos = 2
x = torch.Tensor([1, eos, eos, 3, 3, 3, 4])

print(x == eos)
# the first index of eos
ids = (x == eos).nonzero()
print(ids)
# pad the remainder
x[ids[0] + 1 :] = 0
print(x)
print("---")
# what about a matrix?
x = torch.Tensor([[1, eos, eos, 3, 3, 3, 4], [1, 3, 3, eos, eos, eos, 4]])
# the first index of eos
o = (x == eos).int()
print(o)
o = torch.argsort(o, dim=1)  # this is the trick
print(o)
ids = o[:, -1]  # the index of the first a
# here.. you are effectiely building a mask
# build a mask...
print(ids)
print(x[ids])


# just... how?
print("---")
x = torch.Tensor([[3, 3, 1, 1, 2], [3, 1, 1, 1, 2]])
print(x == 1)
ids = (x == 1).int().argsort(dim=1, descending=True)[:, 0]
print(ids)

y = x.clone()
# okay. but is there a better way than this?
for i, row in enumerate(y):
    # replace tensor with a range
    row[ids[i] + 1 :] = 2
print(y)

# how do I make this by only using batch operations?
answer = torch.Tensor([[3, 3, 1, 2, 2], [3, 1, 2, 2, 2]])

print(torch.equal(y, answer))


print("---")
x = torch.tensor([0, 1]).repeat(2, 1)
