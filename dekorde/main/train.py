from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_I, build_M, build_X, build_Y
from keras_preprocessing.text import Tokenizer
import torch
import wandb
import argparse


def main():
    conf = load_conf()

    # ========== loading conf ========== #
    device = load_device()
    d_model = conf.embed_size
    head_size = conf.heads
    depth = conf.depth
    epochs = conf.epochs
    max_length = conf.max_length
    lr = conf.lr

    # ========== wandb =========== #
    wandb_config = {
        'd_model': d_model,
        'head_size': head_size,
        'depth': depth,
        'epochs': epochs,
        'max_length': max_length,
    }
    run = wandb.init(
        project="dekorde",
        entity="artemisdicotiar",
        config=wandb_config
    )

    # ========== loading data ========== #
    gibberish2kor = load_gibberish2kor()

    gibs = [row[0] for row in gibberish2kor]
    kors = [row[1] for row in gibberish2kor]

    # ========== setting tokenizer ========== #
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=gibs + kors)
    vocab_size = len(tokenizer.word_index.keys())

    # ========== converting raw text to tensor ========== #
    X = build_X(gibs, tokenizer, max_length, device)  # (N, L)
    Y = build_Y(kors, tokenizer, max_length, device)  # (N, L)

    M = build_M(kors, head_size, max_length, device)  # (N, L, L)

    # ========== loading model & opts ========== #
    transformer = Transformer(
        d_model=d_model,
        vocab_size=vocab_size,
        max_length=max_length,
        head_size=head_size,
        depth=depth,
        mask=M
    ).to(device=device)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=lr)

    print('START')
    # wandb.watch(transformer, log_freq=5)
    for epoch in range(epochs):
        loss, acc = transformer.training_step(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch:{epoch}, loss:{loss}, acc: {acc}")

        wandb.log({'loss': loss, 'acc': acc})

    model_artifact = wandb.Artifact(
        "trained-dekorder", type="model",
        description="dekorder, korean review translator",
        metadata=wandb_config
    )

    torch.save(transformer.state_dict(), 'dekorder')
    model_artifact.add_file("dekorder")
    wandb.save("dekorder")

    run.log_artifact(model_artifact)

    run.finish()
    wandb.finish()


if __name__ == '__main__':
    main()
