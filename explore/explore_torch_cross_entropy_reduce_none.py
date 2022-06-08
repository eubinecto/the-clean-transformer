import torch
from torch.nn import functional as F

labels = torch.Tensor([[1, 2, 3], [2, 1, 3], [3, 2, 1]])

predictions = torch.Tensor([[10, 20, 30], [10, 20, 30], [10, 20, 30]])

# this is not just a sum of each batch. The default reduce value is mean.
print(F.cross_entropy(predictions, target=labels))  # (3, 3), (3, 3) -> (1,)

# you can verify this default by doing this.  (3, 3) -> (3,) -> (1,)
print(F.cross_entropy(predictions, labels, reduction="none").mean())

# if you want, you can just sum it over. If you then divide the value by N, you get the same loss.
print(F.cross_entropy(predictions, labels, reduction="sum") / 3)

# but... why is it that the default reduction scheme is to average the loss over the batch? why not just sum it?
# https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the
# A: it is so that what it means by "loss" is normalised to that of one instance, regardless of the batch size
