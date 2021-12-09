"""
Defining the dataset & the datamodule to be used for training
"""
import torch
from typing import Tuple, Optional
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split, Subset
from pytorch_lightning import LightningDataModule
from wandb.sdk.wandb_run import Run
from dekorde.builders import TrainInputsBuilder, LabelsBuilder
from dekorde.fetchers import fetch_jeju2seoul
import os

# to suppress warnings - we just allow parallelism
# https://github.com/kakaobrain/pororo/issues/69#issuecomment-927564132
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DekordeDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        N, _ = self.y.size()
        return N


class Jeju2SeoulDataModule(LightningDataModule):

    def __init__(self, run: Run, config: dict, tokenizer: Tokenizer):
        super().__init__()
        self.run = run
        self.config = config
        self.tokenizer = tokenizer
        # --- to be filled --- #
        self.train: Optional[Subset] = None
        self.val: Optional[Subset] = None
        self.test: Optional[Subset] = None

    def prepare_data(self) -> None:
        jeju2seoul = fetch_jeju2seoul(self.run)
        jejus = [jeju for jeju, _ in jeju2seoul]
        seouls = [seoul for _, seoul in jeju2seoul]
        # build the tensors here
        X = TrainInputsBuilder(self.tokenizer, self.config['max_length'])(srcs=jejus, tgts=seouls)  # (N, L)
        y = LabelsBuilder(self.tokenizer, self.config['max_length'])(tgts=seouls)  # (N, L)
        # build the dataset here
        dataset = DekordeDataset(X, y)
        val_size = int(len(dataset) * self.config["val_ratio"])
        test_size = int(len(dataset) * self.config["test_ratio"])
        train_size = len(dataset) - val_size - test_size
        # split the dataset here
        self.train, self.val, self.test = \
            random_split(dataset, lengths=(train_size, val_size, test_size),
                         generator=torch.Generator().manual_seed(self.config['seed']))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])

    # ignore this
    def predict_dataloader(self):
        pass


# --- add more datamodule as you wish --- #
class Kor2EngDataModule(LightningDataModule):

    def train_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    # ignore this
    def predict_dataloader(self) -> DataLoader:
        pass
