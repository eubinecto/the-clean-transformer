import argparse
from transformers import BertTokenizer
from dekorde.components.transformer import Transformer
from dekorde.data import DekordeDataModule
from dekorde.loaders import load_config, load_seoul2jeju
import pytorch_lightning as pl
import torch
from dekorde.paths import TRANSFORMER_CKPT, TOKENIZER_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
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
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         log_every_n_steps=config['log_every_n_steps'],
                         gpus=torch.cuda.device_count(),
                         enable_checkpointing=False,
                         logger=False)
    # start training transformer
    trainer.fit(model=transformer, datamodule=datamodule)
    # save the model and the tokenizer
    trainer.save_checkpoint(TRANSFORMER_CKPT)
    tokenizer.save_pretrained(TOKENIZER_DIR)


if __name__ == '__main__':
    main()
