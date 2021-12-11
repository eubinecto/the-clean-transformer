
import wandb
from enkorde.fetchers import fetch_tokenizer
from enkorde.paths import WANDB_DIR


def main():
    with wandb.init(entity="eubinecto", project="dekorde", dir=WANDB_DIR) as run:
        tokenizer = fetch_tokenizer(run)
        print(tokenizer.pad_token)
        print(tokenizer.unk_token)
        print(tokenizer.eos_token)
        print(tokenizer.bos_token)


if __name__ == '__main__':
    main()
