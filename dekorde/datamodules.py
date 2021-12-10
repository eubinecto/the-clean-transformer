"""
Defining the dataset & the datamodule to be used for training
"""
import os
import torch
from typing import Tuple, Optional, List
from tokenizers import Tokenizer
from koila import lazy
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from dekorde.builders import TrainInputsBuilder, LabelsBuilder
from dekorde.fetchers import fetch_kor2eng

# to suppress warnings - we just allow parallelism
# https://github.com/kakaobrain/pororo/issues/69#issuecomment-927564132
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class DekordeDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = lazy(X, batch=0)
        self.y = lazy(y, batch=0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        N, _ = self.y.size()
        return N


class Kor2EngDataModule(LightningDataModule):
    name: str = "kor2eng"

    def __init__(self, config: dict, tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        # --- to be downloaded --- #
        self.kor2eng_train: Optional[List[Tuple[str, str]]] = None
        self.kor2eng_val: Optional[List[Tuple[str, str]]] = None
        self.kor2eng_test: Optional[List[Tuple[str, str]]] = None

    def prepare_data(self) -> None:
        self.kor2eng_train, self.kor2eng_val, self.kor2eng_test = fetch_kor2eng()

    def build_dataset(self, src2tgt: List[Tuple[str, str]]) -> Dataset:
        srcs = [src for src, _ in src2tgt]
        tgts = [tgt for _, tgt in src2tgt]
        X = TrainInputsBuilder(self.tokenizer, self.config['max_length'])(srcs=srcs, tgts=tgts)  # (N, L)
        y = LabelsBuilder(self.tokenizer, self.config['max_length'])(tgts=tgts)  # (N, L)
        # to save gpu memory
        return DekordeDataset(X, y)  # noqa

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.build_dataset(self.kor2eng_train), batch_size=self.config['batch_size'],
                          shuffle=self.config['shuffle'], num_workers=self.config['num_workers'])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.build_dataset(self.kor2eng_val), batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.build_dataset(self.kor2eng_test), batch_size=self.config['batch_size'],
                          shuffle=False, num_workers=self.config['num_workers'])

    # ignore this
    def predict_dataloader(self):
        pass


class Kor2EngSmallDataModule(Kor2EngDataModule):
    """
    Just the first 50 sentences are prepared.
    We use this just for testing things out
    """
    name: str = "kor2eng_small"

    def prepare_data(self) -> None:
        kor2eng_train, self.kor2eng_val, self.kor2eng_test = fetch_kor2eng()
        self.kor2eng_train = kor2eng_train[:256]
