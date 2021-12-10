import os
import torch
import wandb
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from tqdm import tqdm

from dekorde.models import Transformer
from dekorde.fetchers import fetch_tokenizer, fetch_config
from dekorde.paths import ROOT_DIR
from dekorde.datamodules import Jeju2SeoulDataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="overfit")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    parser.add_argument("--overfit_batches", type=int, default=0)
    args = parser.parse_args()
    config = fetch_config()['train'][args.ver]
    config.update(vars(args))

    with wandb.init(entity="eubinecto", project="dekorde", config=config) as run:
        # --- fetch a pre-trained tokenizer from wandb -- #
        tokenizer = fetch_tokenizer(run, config['tokenizer'])
        # --- instantiate the model to train --- #
        transformer = Transformer(config['hidden_size'],
                                  tokenizer.get_vocab_size(),  # vocab_size
                                  config['max_length'],
                                  tokenizer.pad_token_id,  # noqa
                                  config['heads'],
                                  config['depth'],
                                  config['dropout'],
                                  config['lr'])
        # --- instantiate the data to train the model with --- #
        datamodule = Jeju2SeoulDataModule(run, config, tokenizer)
        # --- prepare the logger (wandb) and the trainer to use --- #
        logger = WandbLogger(log_model=False)
        trainer = Trainer(fast_dev_run=config['fast_dev_run'],
                          overfit_batches=config['overfit_batches'],
                          max_epochs=config['max_epochs'],
                          log_every_n_steps=config['log_every_n_steps'],
                          gpus=torch.cuda.device_count(),
                          enable_checkpointing=False,
                          logger=logger)
        # --- start training --- #
        trainer.fit(model=transformer, datamodule=datamodule)
        # save them only if the training is properly done
        if not config['fast_dev_run'] and trainer.current_epoch == config['max_epochs'] - 1:
            ckpt_path = os.path.join(ROOT_DIR, "transformer.ckpt")
            trainer.save_checkpoint(ckpt_path)
            artifact = wandb.Artifact(name="transformer", type="model", metadata=config)
            artifact.add_file(ckpt_path)
            run.log_artifact(artifact, aliases=["latest", config['ver']])
            os.remove(ckpt_path)  # make sure you remove it after you are done with uploading it


if __name__ == '__main__':
    main()
