from typing import List, Tuple
import torch


def build_X(gibberish2kor: List[Tuple[str, str]], device: torch.device) -> torch.Tensor:
    srcs = [g for g, _ in gibberish2kor]
    targets = [k for _, k in gibberish2kor]
    pass


# 흠..... build_y는 어떻게 하면 되지? 그리 간단하지는 않을텐데, 음.
# autoregressive prediction을 위해선.. 이걸 어떤식으로든 바꿔야한다.
