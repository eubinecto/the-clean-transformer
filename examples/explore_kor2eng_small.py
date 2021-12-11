from enkorde.datamodules import Kor2EngSmallDataModule
from enkorde.fetchers import fetch_tokenizer, fetch_config


def main():
    model = "transformer_torch"
    ver = "overfit_small"
    config = fetch_config()['train'][model][ver]
    config.update({'num_workers': 2})
    datamodule = Kor2EngSmallDataModule(config, None)  # noqa
    datamodule.prepare_data()
    # --- explore some data --- #
    for pair in datamodule.kor2eng_train:
        print(pair)


if __name__ == '__main__':
    main()
