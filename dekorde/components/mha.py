import math

import torch
import torch.nn.functional as F


class MultiHeadAttentionLayer(torch.nn.Module):
    """
    this could be either masked or not.
    """

    def __init__(self, d_model: int, head_size: int, is_masked: bool):
        super().__init__()
        # hidden size must be divisible by heads.
        assert d_model % head_size == 0

        self.is_masked = is_masked

        self.head_size = head_size
        self.head_dim = d_model // self.head_size
        self.d_model = d_model  # d_model = head_dim * head_size

        # any layers to optimise? - four linear layers in total.
        # 논문에서 하나의 d_model 차원의 KVQ를 쓰는 것보다 h번을 나눠 아래의 방식을 적용하는 것이 좋다고 한다.
        # Q => head_dim, K => head_dim, V => d_v로 linearly project 한다고 한다.
        # output은 d_v차
        # 근데 논문에서 head_dim = d_v = d_model / h = 64로 설정하고 있다. (h: head size)

        # 이렇게 처리하는 건 h개의 헤드를 한번에 처리.
        # ( seq_len, d_model (=head_dim * head_size) ) X ( head_dim * head_size (=d_model), d_model )
        self.W_q = torch.nn.Linear(self.d_model, self.d_model)
        self.W_k = torch.nn.Linear(self.d_model, self.d_model)
        self.W_v = torch.nn.Linear(self.d_model, self.d_model)

        # output 도 결국 같은 모양
        self.W_o = torch.nn.Linear(self.d_model, self.d_model)

    def forward(self,
                H_q: torch.Tensor,
                H_k: torch.Tensor,
                H_v: torch.Tensor,
                M: torch.Tensor = None) -> torch.Tensor:
        """
        :param H_q: (N, L, d_model)
        :param H_k: (N, L, d_model)
        :param H_v: (N,z L, d_model)
        :param M: (???) The mask.
        :return: H_all (N, L, H)
        """

        # ============ pass linear layer ============ #
        Q = self.W_q(H_q)  # (N, L, d_model) * (d_model, d_model) -> (N, L, d_model)
        K = self.W_k(H_k)  # (N, L, d_model) * (d_model, d_model) -> (N, L, d_model)
        V = self.W_v(H_v)  # (N, L, d_model) * (d_model, d_model) -> (N, L, d_model)

        # reshaping (N, L, E) -> (N, L, head_size, head_dim)
        # N: 배치 크기 (seq_len)
        # L: 문장 최대 길이
        Q = Q.reshape(Q.size(0), Q.size(1), self.head_size, self.head_dim)
        K = K.reshape(K.size(0), K.size(1), self.head_size, self.head_dim)
        V = V.reshape(V.size(0), V.size(1), self.head_size, self.head_dim)

        # 각 헤드별 어텐션 결과
        attns = self.scaled_dot_product_attention(Q, K, V, M)

        # 헤드별 결과를 concat
        concated_attns = attns.reshape(Q.size(0), Q.size(1), self.head_size * self.head_dim)

        # 최종 아웃풋 linear layer 통과 결과
        return self.W_o(concated_attns)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor,
                                     M: torch.Tensor = None) -> torch.Tensor:
        """
        :param Q: (N, L, head_size(=d_q), head_dim) (nqhd)
        :param K: (N, L, head_size(=d_k), head_dim) (nkhd)
        :param V: (N, L, head_size(=d_v), head_dim) (nvhd)
        :param M: (???)
        :return: H_all (N, L, head_size, head_dim)
        """

        # 각 문장에 대해, 각 헤드가 내놓는 Q - K 어텐션 결과 필요하다.
        # 그렇기 때문에, qk_attn_score 는 N, head_size,
        qk_attn_score: torch.Tensor = torch.einsum("nqhd,nkhd->nhqk", Q, K) / math.sqrt(self.head_dim)

        # Masking
        if self.is_masked:
            qk_attn_score = qk_attn_score.masked_fill(M == 0, -1e9)

        qk_attn_prob = F.softmax(qk_attn_score, dim=-1)

        # 아웃픗으로 원하는 건
        # 각 문장에 대해, 각 헤드가 도출한 어텐션 행렬에서 도출한 점수를
        # 입력받은 V (밸류)에 곱해서
        # 각 문장에 대한 (쿼리길이(=헤드의 차원), 헤드 개수, 헤드 차원) -> (헤드의 차원, 임베딩 크기)로
        # 각 문장에 대한 최종 어텐션 정보를 리턴.

        # nhqk * nvhd -> nqhd 인데 여기서 v = k이다.
        # 따라서 nhqt,nthd -> nqhd 이다. (v = k = t)

        return torch.einsum('nhqt,nthd->nqhd', qk_attn_prob, V)

