import torch
from torch.nn import functional as F

# (N, L)
N = 3
L = 5
V = 10
labels = torch.randint(low=0, high=V - 1, size=(N, L))

# (N, C, L)
preds = torch.rand(size=(N, V, L)).float()

print(labels)
print(preds)

# scalar value. how is this computed?
print(F.cross_entropy(preds, labels))
# by default, that is calculated by taking the average over each length first, and then over the batch.
print(F.cross_entropy(preds, labels, reduction="none"))
print(F.cross_entropy(preds, labels, reduction="none").mean(dim=-1))  # average over length
print(F.cross_entropy(preds, labels, reduction="none").mean(dim=-1).mean(dim=-1))  # average over batch
# they are essentially the same
print(
    F.cross_entropy(preds, labels).item()
    == F.cross_entropy(preds, labels, reduction="none").mean(dim=-1).mean(dim=-1).item()
)
