
from enkorde.datamodules import Kor2EngSmallDataModule
import wandb

from enkorde.fetchers import fetch_tokenizer, fetch_config
from enkorde.paths import WANDB_DIR


def main():
    ver = "overfit_small"
    config = fetch_config()['train'][ver]
    config.update({'num_workers': 2})

    with wandb.init(entity="eubinecto", project="dekorde", dir=WANDB_DIR) as run:
        tokenizer = fetch_tokenizer(run, config['tokenizer'])
        datamodule = Kor2EngSmallDataModule(config, tokenizer)
        datamodule.prepare_data()
        # --- explore some data --- #
        for pair in datamodule.kor2eng_train:
            print(pair)
        # --- explore the tensors --- #
        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            X, y = batch
            src_ids = X[:, 0, 0]
            tgt_ids = X[:, 1, 0]
            print([tokenizer.id_to_token(src_id) for src_id in src_ids[0].tolist()])
            print([tokenizer.id_to_token(tgt_id) for tgt_id in tgt_ids[0].tolist()])
            # should be right-shifted
            print([tokenizer.id_to_token(y_id) for y_id in y[0].tolist()])
            print("-----")


if __name__ == '__main__':
    main()
