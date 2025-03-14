"""
Note: I did not write this code. This code was made publically available at https://github.com/mjliu2020/RandomFR/. 
It is used to load the datasets. 
- Tyler
"""

"""
Second Note: In order to get accurate results here, make sure that the 'fc_matrix_path' and 'position_path' variables are set correctly
to match where your data is stored.
"""

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import heapq


class GDataset(Dataset):

    def __init__(self, idxs, csv):
        super(GDataset, self).__init__()
        self.fc_matrix_dot = os.path.join('data')
        self.idxs = idxs
        self.csv = csv

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        file_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(np.int)

        fc_matrix_path = os.path.join(self.fc_matrix_dot, 'FCmatrix', str(file_name), f'fcmatrix.npy')
        fc_matrix0 = np.load(fc_matrix_path).astype(np.float32)
        fc_matrix = np.abs(fc_matrix0)

        position_path = f'./data/subjectwise_position/{file_name}.npy'
        position = np.load(position_path).astype(np.float32)

        return torch.tensor(fc_matrix[:256, :]), torch.tensor(label), torch.tensor(position[:256, :])


def get_data_loader(i_fold):

    df = pd.read_csv('data/NYU_5fold.csv')

    df = df.reset_index()
    train_idxs = np.where(df['fold'] != i_fold)[0]
    test_idxs = np.where(df['fold'] == i_fold)[0]

    TrainDataset = GDataset(train_idxs, df)
    TestDataset = GDataset(test_idxs, df)
    TrainLoader = DataLoader(TrainDataset, batch_size=16, shuffle=True, drop_last=True)
    TestLoader = DataLoader(TestDataset, batch_size=1)
    return TrainLoader, TestLoader


if __name__ == '__main__':
    TrainLoader, TestLoader = get_data_loader(0)
    for fc, crs, posi in TrainLoader:
        print(fc.shape, crs.shape, posi.shape)
        break
    print(' ')
    for fc, crs, posi in TestLoader:
        print(fc.shape, crs.shape, posi.shape)
        break
