import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split, Subset
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizer
from wandb.sdk.wandb_run import Run
from dekorde.builders import TrainInputsBuilder, LabelsBuilder


class DekordeDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        N, _ = self.y.size()
        return N


class DekordeDataModule(LightningDataModule):

    def __init__(self, run: Run, config: dict, tokenizer: BertTokenizer):
        super().__init__()
        self.run = run
        self.config = config
        self.tokenizer = tokenizer
        self.seoul2jeju: Optional[List[Tuple[str, str]]] = None
        self.train: Optional[Subset] = None
        self.val: Optional[Subset] = None
        self.test: Optional[Subset] = None

    def prepare_data(self) -> None:
        # download the data
        artifact = self.run.use_artifact(f"seoul2jeju:{self.config['data_ver']}")
        table = artifact.get("seoul2jeju")
        self.seoul2jeju = [(row[0], row[1]) for row in table.data]

    def setup(self, *args, **kwargs) -> None:
        seouls = [seoul for seoul, _ in self.seoul2jeju]
        jejus = [jeju for _, jeju in self.seoul2jeju]
        # build the tensors here
        X = TrainInputsBuilder(self.tokenizer, self.config['max_length'])(srcs=jejus, tgts=seouls)  # (N, L)
        y = LabelsBuilder(self.tokenizer, self.config['max_length'])(tgts=seouls)  # (N, L)
        # build the dataset here
        dataset = DekordeDataset(X, y)
        val_size = int(len(dataset) * self.config["val_ratio"])
        test_size = int(len(dataset) * self.config["test_ratio"])
        train_size = len(dataset) - val_size - test_size
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

    def predict_dataloader(self) -> DataLoader:
        pass
