import torch  # noqa


class FeedForward(torch.nn.Module):
    """
    position-wise feedforward network.
    """

    def __init__(self, hidden_size: int, ffn_size: int, dropout: float):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_size),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_size, hidden_size),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_size),
        )

    def forward(self, x: torch.Tensor):
        """
        :param x: (N, L, H)
        :return: x (hidden): (N, L, H)
        """
        return self.layers(x)
