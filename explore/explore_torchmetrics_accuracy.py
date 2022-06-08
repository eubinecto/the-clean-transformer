import torch
from torchmetrics import functional as F

N = 2
L = 3
V = 4


# (N, L)
labels = torch.Tensor([[0, 1, 2], [2, 0, 1], [2, 1, 0]]).int()

# (N, L, V)
predictions = torch.Tensor(
    [
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
        [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
    ]
)
# (N, L, V) -> (N, V, L) - this is what the accuracy metric requires
predictions = predictions.transpose(1, 2)
print(F.accuracy(predictions, labels))
