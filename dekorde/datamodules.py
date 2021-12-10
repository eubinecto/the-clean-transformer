"""
Defining the dataset & the datamodule to be used for training
"""
import torch
from typing import Tuple, Optional, List
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from wandb.sdk.wandb_run import Run
from dekorde.builders import TrainInputsBuilder, LabelsBuilder
from dekorde.fetchers import fetch_kor2eng
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


class Kor2EngDataModule(LightningDataModule):

    def __init__(self, run: Run, config: dict, tokenizer: Tokenizer):
        super().__init__()
        self.run = run
        self.config = config
        self.tokenizer = tokenizer
        # --- to be filled --- #
        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        kor2eng_train, kor2eng_val, kor2eng_test = fetch_kor2eng()
        # split the dataset here
        self.train = self.build_dataset(kor2eng_train)
        self.val = self.build_dataset(kor2eng_val)
        self.test = self.build_dataset(kor2eng_test)

    def build_dataset(self, src2tgt: List[Tuple[str, str]]) -> Dataset:
        srcs = [src for src, _ in src2tgt]
        tgts = [tgt for _, tgt in src2tgt]
        X = TrainInputsBuilder(self.tokenizer, self.config['max_length'])(srcs=srcs, tgts=tgts)  # (N, L)
        y = LabelsBuilder(self.tokenizer, self.config['max_length'])(tgts=tgts)  # (N, L)
        return DekordeDataset(X, y)

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
