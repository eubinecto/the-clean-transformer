from dekorde.components.transformer import Transformer
from dekorde.loaders import load_conf, load_device, load_gibberish2kor
from dekorde.builders import build_I, build_M
from keras_preprocessing.text import Tokenizer
import torch
import wandb


def train():
    with wandb.init(
            project="dekorde",
            entity="artemisdicotiar",
    ) as run:

        # ========== loading conf ========== #
        conf = wandb.config
        device = load_device()

        d_model = conf['d_model']
        head_size = conf['head_size']
        depth = conf['depth']
        epochs = conf['epochs']
        max_length = conf['max_length']
        lr = conf['lr']

        # ========== loading data ========== #
        gibberish2kor = load_gibberish2kor()

        gibs = [row[0] for row in gibberish2kor]
        kors = [row[1] for row in gibberish2kor]

        # ========== setting tokenizer ========== #
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(texts=gibs + kors)
        vocab_size = len(tokenizer.word_index.keys())

        # ========== converting raw text to tensor ========== #
        X = build_I(gibs, tokenizer, max_length, device)  # (N, L)
        Y = build_I(kors, tokenizer, max_length, device)  # (N, L)

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


def main():
    # ========== wandb =========== #
    sweep_config = {
        "name": "dekorde-sweep",
        "program": train,
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "acc"
        },
        "parameters": {
            "max_length": {
                "value": 30,
            },
            "head_size": {
                "max": 16,
                "min": 4,
                "distribution": 'int_uniform',
            },
            "d_model": {
                "value": 64
            },
            "epochs": {
                "max": 300,
                "min": 75,
                "distribution": 'int_uniform',
            },
            "depth": {
                "max": 6,
                "min": 2,
                "distribution": 'int_uniform',
            },
            "lr": {
                "value": 0.0001
            }
        }
    }
    sweep = wandb.sweep(sweep_config, project='dekorde', entity='artemisdicotiar')
    count = 4  # number of runs to execute

    wandb.agent(sweep, function=train, count=count)


if __name__ == '__main__':
    main()
