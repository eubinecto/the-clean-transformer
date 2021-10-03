import torch


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass


class Encoder(torch.nn.Module):
    def __init__(self, depth: int):
        self.depth = depth
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass
