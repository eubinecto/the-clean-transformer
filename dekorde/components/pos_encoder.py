import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout_rate: int = 0,
                 max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # base dropout rate is set as 0. => fully wired.
        self.dropout = nn.Dropout(p=dropout_rate)

        # ======= calculate positional encoding ======= #
        # pos_encoding 행렬 세팅
        pos_enc = torch.zeros(max_len, d_model)     # (max_len, d_model) -> all zeros

        # 0 부터 max_len까지 주루륵
        position = torch.arange(0, max_len)         # (1, max_len)
        position = position.unsqueeze(1)            # (1, max_len) -> (max_len, 1)

        # 삼각함수 내의 각도 구하기
        angle = position / torch.pow(
            input=torch.tensor(10000.0),
            exponent=torch.arange(0, d_model, 2) / d_model
        )

        pos_enc[:, 0::2] = torch.sin(angle)         # 짝수번째 칼럼만 값 업데이트 [0::2]
        pos_enc[:, 1::2] = torch.cos(angle)         # 홀수번째 칼럼만 값 업데이트  [1::2]

        self.positional_encoding = pos_enc.unsqueeze(0)

    def forward(self, x):
        out = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        return self.dropout(out)
