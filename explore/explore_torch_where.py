import torch


x = torch.randint(high=5, low=1, size=(10, 7))
print(x)
for t in range(x.shape[1] - 1):
    x[:, t + 1] = torch.where(x[:, t] == 2, 2, x[:, t + 1])

print(x)
