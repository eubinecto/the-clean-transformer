import torch
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer
from dekorde.components.transformer import Transformer
from dekorde.data import DekordeDataModule
from dekorde.loaders import load_config
import pytorch_lightning as pl
from dekorde.paths import transformer_paths


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
    # save the model and the tokenizer
    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        datamodule = DekordeDataModule(run, config, tokenizer)
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             log_every_n_steps=config['log_every_n_steps'],
                             gpus=torch.cuda.device_count(),
                             enable_checkpointing=False,
                             logger=logger)
        # start training transformer
        trainer.fit(model=transformer, datamodule=datamodule)
    # save them only if the training is properly done
    if trainer.current_epoch == config['max_epochs']:
        transformer_ckpt, tokenizer_dir = transformer_paths()
        trainer.save_checkpoint(transformer_ckpt)
        tokenizer.save_pretrained(tokenizer_dir)


if __name__ == '__main__':
    main()
