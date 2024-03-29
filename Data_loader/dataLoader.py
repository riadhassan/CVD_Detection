import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


class CVD_dataloader(Dataset):

    def __init__(
            self,
            data_dir,
            subset="train",
            ids = None
    ):
        self.subset = subset
        self.ids = ids
        self.data_dir = data_dir

        self.data = pd.read_csv(self.data_dir, sep=";")
        print(self.subset)
        print(np.array(self.ids).min())
        print(np.array(self.ids).max())

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, id):
        row = self.data.values[id]

        row_param = torch.from_numpy(np.array(row[1:-1]))
        row_gt = torch.from_numpy(np.array(row[-1]))

        return row_param, row_gt


def data_loaders(train_ids, test_ids, data_path):
    dataset_train, dataset_valid = datasets(train_ids, test_ids, data_path)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=0,
        worker_init_fn=worker_init,
        sampler=train_ids
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        num_workers=0,
        worker_init_fn=worker_init,
        sampler= test_ids
    )

    return loader_train, loader_valid


def datasets(train_ids, test_ids, data_path):
    train = CVD_dataloader(
        data_dir=data_path,
        subset="train",
        ids = train_ids
    )
    valid = CVD_dataloader(
        data_dir=data_path,
        subset="validation",
        ids = test_ids
    )
    return train, valid