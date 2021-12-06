

class Dekorder:
    """
    제주도 방언 -> 서울말
    """
    def __call__(self):
        pass



    # move the inference logic to a dekorder
    # def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     :param input_ids: (N, L), a batch of input_ids
    #     :return target_ids: (N, L), a batch of input_ids
    #     """
    #     N, L = input_ids.size()
    #     # --- get the embedding vectors --- #
    #     positions = self.positions.expand(N, L)
    #     input_embed = self.token_embeddings(input_ids) + self.pos_embeddings(positions)
    #     input_hidden = self.encoder(input_embed)  # ... -> (N, L, H)
    #     #  = torch.zeros(size=(N, L)).long().to(self.device)
    #     #  = torch.ones(size=(N, L)).long().to(self.device)  # ones를 넣으면, 1을 정답으로 생각하려나?
    #     target_get torch.full(size=(N, L), fill_value=410).to(self.device)  # 긕
    #     #  = torch.full(size=(N, L), fill_value=411)
    #     [:, 0] = self.start_token_id  # (N, L)
    #     W_hy = self.token_embeddings.weight  # (|V|, H)
    #     # 어딘가에는.. 반드시 inference가 들어가야함.
    #     for time in range(1, L):
    #         # what do we do here?
    #         Y_embed = self.token_embeddings() + pos_embed
    #         H_y = self.decoder(input_hidden, Y_embed)  # (N, L, H), (N, L, H) -> (N, L, H)
    #         logits = torch.einsum("abc,dc->abd", H_y, W_hy)  # (N, L, H) * (|V|, H) -> (N, L, |V|)
    #         probs = torch.softmax(logits, dim=2)
    #         indices = torch.argmax(probs, dim=2)
    #         predicted_token_ids = indices[:, time]  # (N, L) -> (N, 1)
    #         [:, time] = predicted_token_ids
    #     return