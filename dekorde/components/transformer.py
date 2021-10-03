import torch


class Transformer(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass
