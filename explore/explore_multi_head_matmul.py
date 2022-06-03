import torch


def main():
    N = 10
    L = 30
    heads = 8
    H = 64
    q = torch.rand(size=(N, L, H))  # (N, heads, L, H)
    k = torch.rand(size=(N, L, H))  # (N, heads, L, H)
    v = torch.rand(size=(N, L, H))  # (N, heads, L, H)
    # --- split this into multi-heads --- #
    q = q.view(N, L, heads, H // heads)
    k = k.view(N, L, heads, H // heads)
    v = v.view(N, L, heads, H // heads)
    # --- to make them matmul-compatible --- #
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # --- using torch.matmul --- #
    sims = q @ k.transpose(2, 3)
    attentions = torch.softmax(sims, dim=-1)
    alignments = attentions @ v
    concats = alignments.transpose(1, 2).contiguous().view(N, L, H)
    print(concats.shape)
    # --- using einsum with elipsis --- #
    # this way, we can apply this to both single-headed and multi-headed attention
    sims = torch.einsum("...qh,...kh->...qk", q, k)  # this way, you don't need to call "k.transpose"
    attentions = torch.softmax(sims, dim=-1)
    alignments = torch.einsum("...qk,...kh->...qh", attentions, v)
    concats = alignments.transpose(1, 2).contiguous().view(N, L, H)
    print(concats.shape)


if __name__ == "__main__":
    main()
