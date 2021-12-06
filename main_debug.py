import argparse
from transformers import BertTokenizer
from dekorde.components.transformer import Transformer
from dekorde.data import DekordeDataModule
from dekorde.loaders import load_config, load_seoul2jeju
import pytorch_lightning as pl
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config()
    config.update(vars(args))
    # --- load a tokenizer --- #
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer'])  # pre-trained on colloquial data
    tokenizer.add_special_tokens({'bos_token': config['bos_token'],
                                  'eos_token': config['eos_token']})
    # --- instantiate the model and the optimizer --- #
    transformer = Transformer(config['hidden_size'],
                              len(tokenizer),  # vocab_size
                              config['max_length'],
                              config['heads'],
                              config['depth'],
                              config['dropout'],
                              tokenizer.pad_token_id,
                              config['lr'])
    datamodule = DekordeDataModule(config, tokenizer)
    # save the model and the tokenizer
    trainer = pl.Trainer(fast_dev_run=True,  # 에폭을 한번만 돈다. 모델 저장도 안함. 디버깅으로 제격
                         gpus=torch.cuda.device_count())
    # start training transformer
    trainer.fit(model=transformer, datamodule=datamodule)


if __name__ == '__main__':
    main()
