import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
indices = torch.tensor([0, 1, 0, 1]).unsqueeze(dim=-1)
print(a.gather(1, indices))
