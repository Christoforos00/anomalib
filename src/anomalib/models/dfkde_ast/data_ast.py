import io
import os
import pickle

import torch

import pandas as pd

from typing import Dict, Optional, Tuple, Union

from anomalib.config import get_configurable_parameters
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    # elif torch.backends.mps.is_available():
    #     return "mps:0"
    else:
        return "cpu"


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class PickleDataset(LightningDataModule):

    def __init__(self, data_dir, file_2_label, files_list=None):
        super().__init__()

        self.data_dir = data_dir
        self.files_list = [f for f in os.listdir(self.data_dir) if ".pickle" in f]

        if files_list is not None:
            self.files_list = [file for file in self.files_list if file in files_list]

        self._len = len(self.files_list)
        self.file_2_label = file_2_label

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        file_name = self.files_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "rb") as input_file:
            features = CPU_Unpickler(input_file).load()
        label = self.file_2_label[file_name]
        item = {"image": features, "label": label, "file_name": file_name}
        return item


class CustomDataModule(LightningDataModule):

    def __init__(
            self,
            batch_size,
            data_dir,
            labels_dir,
            num_workers: int = 8,
            seed: Optional[int] = 101,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        labels_df = pd.read_csv(self.labels_dir)
        file_2_label = dict(zip(labels_df["names"], labels_df["annotation_label"]))

        test_files = labels_df[labels_df["is_annotated"] == 1]["names"].tolist()
        train_files = [file for file in labels_df["names"].tolist() if file not in test_files]

        whoop_files = []
        blank_files = []

        for file in train_files:
            if file_2_label[file]:
                blank_files.append(file)
            else:
                whoop_files.append(file)

        idx = int(len(train_files) * 0.5)
        train_files = whoop_files[:idx]
        val_files = blank_files
        val_files.extend(whoop_files[idx:])

        self.train_data = PickleDataset(self.data_dir, file_2_label, files_list=train_files)
        self.val_data = PickleDataset(self.data_dir, file_2_label, files_list=val_files)
        self.test_data = PickleDataset(self.data_dir, file_2_label, files_list=test_files)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_data,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    from importlib import import_module


    def _snake_to_pascal_case(model_name: str) -> str:
        """Convert model name from snake case to Pascal case.

        Args:
            model_name (str): Model name in snake case.

        Returns:
            str: Model name in Pascal case.
        """
        return "".join([split.capitalize() for split in model_name.split("_")])


    MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    CONFIG_PATH = "config.yaml"
    # with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
    #     print(file.read())

    # pass the config file to model, callbacks and datamodule
    config = get_configurable_parameters(config_path=CONFIG_PATH)

    name = "dfkde_ast"

    module = import_module(f"anomalib.models.{name}")
    model = getattr(module, f"{_snake_to_pascal_case(name)}Lightning")(config)
