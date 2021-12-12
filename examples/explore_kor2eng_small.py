from enkorde.datamodules import Kor2EngSmallDataModule
from enkorde.fetchers import fetch_tokenizer, fetch_config


def main():
    ver = "overfit_small"
    config = fetch_config()['train'][ver]
    config.update({'num_workers': 2})
    datamodule = Kor2EngSmallDataModule(config, None)  # noqa
    datamodule.prepare_data()
    # --- explore some data --- #
    for pair in datamodule.kor2eng_train:
        print(pair)


if __name__ == '__main__':
    main()
