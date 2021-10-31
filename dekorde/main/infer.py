from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_seoul2jeju
from dekorde.builders import XBuilder, YBuilder, build_lookahead_mask
from transformers import BertTokenizer
import torch


def main():
    conf = load_conf()
    # -- conf; hyper parameters --- #
    device = load_device()
    max_length = conf['max_length']
    hidden_size = conf['hidden_size']
    batch_size = conf['batch_size']  # batch processing to be implemented later.
    heads = conf['heads']
    depth = conf['depth']
    lr = conf['lr']

    # --- build the data --- #
    seoul2jeju = load_seoul2jeju()[:50]
    seouls = [seoul for seoul, _ in seoul2jeju]
    jejus = [jeju for _, jeju in seoul2jeju]

    # --- load a tokenizer --- #
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")  # pre-trained on colloquial data
    start_token = "[SOS]"
    tokenizer.add_tokens("[SOS]")  # add the start-of-sequence token.
    start_token_id = tokenizer.convert_tokens_to_ids(start_token)
    X = XBuilder(tokenizer, max_length, start_token, device)(seouls)  # (N, L)
    Y = YBuilder(tokenizer, max_length, start_token, device)(jejus)  # (N, 2, L)
    lookahead_mask = build_lookahead_mask(max_length, device)  # (L, L)

    # --- instantiate the model and the optimizer --- #
    transformer = Transformer(hidden_size, len(tokenizer), max_length, heads, depth,
                              start_token_id, lookahead_mask, device)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=lr)

    # --- start training --- #
    for epoch in range(conf['epochs']):
        # compute the loss.
        loss, acc = transformer.training_step(X, Y)
        loss.backward()  # backprop
        optimizer.step()  # gradient descent
        optimizer.zero_grad()  # prevent the gradients accumulating.
        print(f"epoch:{epoch}, loss:{loss}, acc: {acc}")

    # you may want to save the model & the tokenizer as well
    Y_preds = transformer.infer(X[7:8])

    infers = [tokenizer.convert_ids_to_tokens(Y_pred) for Y_pred in Y_preds]
    print(infers)


if __name__ == '__main__':
    main()
