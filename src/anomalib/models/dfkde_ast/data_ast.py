import io
import os
import pickle

import librosa
import numpy as np
import torch

import pandas as pd

from typing import Optional

import torchaudio

from anomalib.config import get_configurable_parameters
from torch.utils.data import DataLoader
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


def resample_if_necessary(signal, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        signal = torch.tensor(signal).to(torch.float)
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        signal = resampler(signal).detach().cpu().numpy()
    return signal


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


class PickleDataset(LightningDataModule):

    def __init__(self, data_dir, file_2_label, files_list=None):
        super().__init__()

        self.data_dir = data_dir
        self.files_list = [f for f in os.listdir(self.data_dir) if ".pickle" in f]

        if files_list is not None:
            self.files_list = [file for file in files_list if file in self.files_list]

        self._len = len(self.files_list)
        self.file_2_label = file_2_label

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        file_name = self.files_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        with open(file_path, "rb") as input_file:
            features = CPU_Unpickler(input_file).load()
        if features.dim() == 2:
            device = features.device
            features = mono_to_color(features.detach().cpu().numpy())
            features = torch.tensor(features, dtype=torch.float32).permute(2, 0, 1).to(device)
        label = self.file_2_label[file_name]
        item = {"image": features, "label": label, "file_name": file_name}
        return item


class SimpleAudioDataset(LightningDataModule):
    def __init__(self, data_dir, file_2_label, file_seconds=None, target_sample_rate=16000, files_list=None):
        self.data_dir = data_dir
        self.sample_size = target_sample_rate * file_seconds if file_seconds else 0
        self.target_sample_rate = target_sample_rate
        self.file_2_label = file_2_label
        self.files_list = [f for f in os.listdir(self.data_dir) if f[0] != "."]
        if files_list is not None:
            self.files_list = [file for file in self.files_list if file in files_list]
        self._len = len(self.files_list)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        filename = self.files_list[idx]
        audio_sample_path = os.path.join(self.data_dir, filename)
        y, sr = librosa.load(audio_sample_path, sr=None)
        y = resample_if_necessary(y, sr, self.target_sample_rate)
        if self.sample_size:
            y = np.pad(y, (0, self.sample_size - y.shape[0]), 'constant')
        label = self.file_2_label[filename]
        item = {"image": y, "label": label, "file_name": filename}
        return item


class CustomDataModule(LightningDataModule):

    def __init__(
            self,
            batch_size,
            data_dir,
            file_2_label,
            train_files,
            val_files,
            test_files,
            raw_audio: bool = False,
            num_workers: int = 8,
            seed: Optional[int] = 101,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.file_2_label = file_2_label
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.raw_audio = raw_audio
        self.train_data = None
        self.val_data = None
        self.test_data = None
        #
        # self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if self.raw_audio:
            self.train_data = SimpleAudioDataset(self.data_dir, self.file_2_label, files_list=self.train_files)
            self.val_data = SimpleAudioDataset(self.data_dir, self.file_2_label, files_list=self.val_files)
            self.test_data = SimpleAudioDataset(self.data_dir, self.file_2_label, files_list=self.test_files)
            self.predict_data = SimpleAudioDataset(self.data_dir, self.file_2_label,
                                                   files_list=self.train_files + self.val_files + self.test_files)
        else:
            self.train_data = PickleDataset(self.data_dir, self.file_2_label, files_list=self.train_files)
            self.val_data = PickleDataset(self.data_dir, self.file_2_label, files_list=self.val_files)
            self.test_data = PickleDataset(self.data_dir, self.file_2_label, files_list=self.test_files)
            self.predict_data = PickleDataset(self.data_dir, self.file_2_label,
                                              files_list=self.train_files + self.val_files + self.test_files)

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

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.predict_data,
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
