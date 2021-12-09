
import wandb

from fetchers import fetch_tokenizer


def main():
    with wandb.init(entity="eubinecto", project="dekorde") as run:
        tokenizer = fetch_tokenizer(run)
        print(tokenizer.padding)


if __name__ == '__main__':
    main()
