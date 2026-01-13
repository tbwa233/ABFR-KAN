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
        label = self.csv['label'].iloc[index].astype(int)

        # Make sure the paths in the two lines beneath this comment match the data you are expecting to use
        fc_matrix_path = os.path.join(self.fc_matrix_dot, 'FCmatrix_random_iterative', str(file_name), f'fcmatrix.npy')
        position_path = f'./data/subjectwise_position_random_iterative/{file_name}.npy'
        
        fc_matrix0 = np.load(fc_matrix_path).astype(np.float32)
        fc_matrix = np.abs(fc_matrix0)
        position = np.load(position_path).astype(np.float32)

        return torch.tensor(fc_matrix[:256, :]), torch.tensor(label), torch.tensor(position[:256, :])

def get_data_loader(i_fold):

    df = pd.read_csv('data/UM_5fold.csv') # Make sure the .csv used here matches with the fcmatrix and subjectwise_position files you are using

    df = df.reset_index()
    train_idxs = np.where(df['fold'] != i_fold)[0]
    val_idxs = np.where(df['fold'] == i_fold)[0]

    TrainDataset = GDataset(train_idxs, df)
    ValDataset = GDataset(val_idxs, df)
    TrainLoader = DataLoader(TrainDataset, batch_size=16, shuffle=True, drop_last=True)
    ValLoader = DataLoader(ValDataset, batch_size=1)
    return TrainLoader, ValLoader

if __name__ == '__main__':
    TrainLoader, ValLoader = get_data_loader(0)
    for fc, crs, posi in TrainLoader:
        print(fc.shape, crs.shape, posi.shape)
        break
    print(' ')
    for fc, crs, posi in ValLoader:
        print(fc.shape, crs.shape, posi.shape)
        break
