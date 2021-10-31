from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_seoul2jeju
from dekorde.builders import XBuilder, YBuilder, build_mask
from transformers import BertTokenizer
import torch


def main():
    conf = load_conf()
    # -- conf; hyper parameters --- #
    device = load_device()
    print(device)
    max_length = conf['max_length']
    hidden_size = conf['hidden_size']
    batch_size = conf['batch_size']  #  batch processing to be implemented later.
    heads = conf['heads']
    depth = conf['depth']
    lr = conf['lr']

    # --- build the data --- #
    seoul2jeju = load_seoul2jeju()[:10]
    seouls = [seoul for seoul, _ in seoul2jeju]
    jejus = [jeju for _, jeju in seoul2jeju]

    # --- load a tokenizer --- #
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")  # pre-trained on colloquial data
    start_token = "[SOS]"
    tokenizer.add_tokens("[SOS]")  # add the start-of-sequence token.
    start_token_id = tokenizer.convert_tokens_to_ids(start_token)
    X = XBuilder(tokenizer, max_length, device)(seouls)  # (N, L)
    Y = YBuilder(tokenizer, max_length, start_token, device)(jejus)  # (N, 2, L)
    padding_mask = build_mask(X, device, option='padding')
    lookahead_mask = build_mask(X, device, option='lookahead')  # (L, L)

    # --- instantiate the model and the optimizer --- #
    transformer = Transformer(hidden_size, len(tokenizer), max_length, heads, depth,
                              start_token_id, padding_mask, lookahead_mask, device)
    optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=lr)

    # --- start training --- #
    for epoch in range(conf['epochs']):
        # compute the loss.
        loss = transformer.training_step(X, Y)
        loss.backward()  # backprop
        optimizer.step()  # gradient descent
        optimizer.zero_grad()  # prevent the gradients accumulating.
        print(f"epoch:{epoch}, loss:{loss}")

    # you may want to save the model & the tokenizer as well
    Y_pred = transformer.infer(X)
    print("="*100)
    for ids in Y_pred:
        print(" ".join(tokenizer.convert_ids_to_tokens(ids)), end="\n")

if __name__ == '__main__':
    main()
