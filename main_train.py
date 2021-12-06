import torch
from torch.optim.lr_scheduler import ExponentialLR
from transformers import BertTokenizer
from dekorde.components.transformer import Transformer
from dekorde.loaders import load_config, load_device, load_seoul2jeju
from dekorde.tensors import InputsBuilder, TargetsBuilder


def main():
    config = load_config()
    device = load_device()
    # --- build the data --- #
    seoul2jeju = load_seoul2jeju()[:5]
    seouls = [seoul for seoul, _ in seoul2jeju]
    jejus = [jeju for _, jeju in seoul2jeju]
    # --- load a tokenizer --- #
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer'])  # pre-trained on colloquial data
    tokenizer.add_special_tokens({'bos_token': config['bos_token'],
                                  'eos_token': config['eos_token']})
    # you may as well... just add a start token here.
    inputs = InputsBuilder(tokenizer, config['max_length'])(seouls)  # (N, L)
    targets = TargetsBuilder(tokenizer, config['max_length'])(jejus)  # (N, 2, L)
    # --- instantiate the model and the optimizer --- #
    transformer = Transformer(config['hidden_size'],
                              len(tokenizer),  # vocab_size
                              config['max_length'],
                              config['heads'],
                              config['depth'],
                              config['dropout'])
    # --- initialise the weights --- #
    for param in transformer.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    # --- register the weights to an optimizer --- #
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=config['lr'])
    # scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    # --- load the data & the model to device (we need this for parallelization) --- #
    inputs.to(device)
    targets.to(device)
    transformer.to(device)
    # --- start training --- #
    for epoch in range(config['epochs']):
        # compute the loss.
        loss = transformer.training_step(inputs, targets)
        loss.backward()  # backpropagating the gradient of loss with respect to all the weights
        optimizer.step()  # taking a gradient descent step
        optimizer.zero_grad()  # so that we prevent gradients from accumulating
        # scheduler.step()
        print(f"epoch:{epoch}, loss:{loss}")

    # # you may want to save the model & the tokenizer as well
    # Y_pred = transformer.infer(X)
    # print(Y_pred)
    # for row in Y_pred.tolist():
    #     print(tokenizer.decode(row))


if __name__ == '__main__':
    main()
