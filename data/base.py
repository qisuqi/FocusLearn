from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from AttnAM_PyTorch.data.processing import ProcessData
from torch.utils.data import Subset, random_split


class CSVDataset(torch.utils.data.Dataset):

    def __init__(self,
                 config,
                 data: pd.DataFrame,
                 features_columns: list,
                 targets_column: str):

        """Custom dataset for csv files.

        Args:
            config ([type]): [description]
            data (str): [description]
            features_columns (list): [description]
            targets_column (str): [description]
        """

        self._config = config
        self.data = data

        self.new_data = self.get_train_val_test()

        self.features_columns = features_columns
        self.targets_column = targets_column

        self.raw_X = self.new_data[features_columns].copy()
        self.raw_y = self.new_data[targets_column].copy()

        self.X, self.y = self.raw_X.to_numpy(), self.raw_y.to_numpy()

    def get_train_val_test(self):

        """ Use function split_data_per_user to get training, validation, and testing set per users, then combine
        all sets together. """

        get_data = ProcessData()

        if 'Oticon' or 'Life' in self.config.name:
            train_x, val_x, test_x, train_y, val_y, test_y = \
                get_data.split_dataset_per_user(self.data)

            self.train = pd.concat((train_x, train_y['Outcome']), axis=1)
            val = pd.concat((val_x, val_y['Outcome']), axis=1)
            self.test = pd.concat((test_x, test_y['Outcome']), axis=1)

        else:
            self.data = self.data.drop('ID', axis=1)
            if self.config.regression:
                train_x, val_x, test_x, train_y, val_y, test_y = \
                     get_data.random_split(self.data)

            else:
                if self.config.resample:
                    train_x, val_x, test_x, train_y, val_y, test_y = \
                        get_data.stratify_split(self.data, resample=True)
                else:
                    train_x, val_x, test_x, train_y, val_y, test_y = \
                        get_data.stratify_split(self.data, resample=False)

            self.train = pd.concat((train_x, train_y), axis=1)
            val = pd.concat((val_x, val_y), axis=1)
            self.test = pd.concat((test_x, test_y), axis=1)

        self.train_size = len(self.train)
        self.val_size = len(val)
        self.test_size = len(self.test)

        new_data = pd.concat((self.train, val, self.test), axis=0)

        return new_data

    def __len__(self):
        return len(self.new_data)
        # return len(self.data)

    @property
    def config(self):
        return self._config


class AttnAMDataset(CSVDataset):

    def __init__(self,
                 config,
                 data: pd.DataFrame,
                 features_columns: list,
                 targets_column: str) -> None:

        super().__init__(config=config,
                         data=data,
                         features_columns=features_columns,
                         targets_column=targets_column)

        self.features, self.features_names = self.raw_X.values, self.raw_X.columns
        self.features = self.features.astype('float32')

        if self.config.regression:
            self.y = self.y.astype('float32')
        else:
            self.y = self.y.astype('int')

        self.compute_features()

        self.features = torch.from_numpy(self.features).float().to(config.device)
        self.targets = torch.from_numpy(self.y).view(-1, 1).float().to(config.device)

        self.setup_dataloaders()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx]

    def compute_features(self):
        single_features = np.split(np.array(self.features), self.features.shape[1], axis=1)
        self.unique_features = [np.unique(f, axis=0) for f in single_features]

        self.single_features = {col: sorted(self.raw_X[col].to_numpy()) for col in self.raw_X}
        self.ufo = {col: sorted(self.raw_X[col].unique()) for col in self.raw_X}

    def setup_dataloaders(self):

        # test_size = int(0.2 * len(self))
        # val_size = int(0.2 * (len(self) - test_size))
        # train_size = len(self) - val_size - test_size
        #
        # train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])

        train_subset = Subset(self, range(0, self.train_size))
        val_subset = Subset(self, range(self.train_size, self.train_size + self.val_size))
        test_subset = Subset(self, range(self.train_size + self.val_size, self.train_size + self.val_size + self.test_size))

        self.train_dl = DataLoader(train_subset,
                                   batch_size=self.config.batch_size,
                                   shuffle=True,
                                   num_workers=self.config.num_workers)
        self.val_dl = DataLoader(val_subset,
                                 batch_size=self.config.batch_size,
                                 shuffle=False,
                                 num_workers=self.config.num_workers)
        self.test_dl = DataLoader(test_subset,
                                  batch_size=self.config.batch_size,
                                  shuffle=False,
                                  num_workers=self.config.num_workers)

    def train_dataloaders(self) -> Tuple[DataLoader, ...]:
        return self.train_dl, self.val_dl

    def test_dataloaders(self) -> DataLoader:
        return self.test_dl
