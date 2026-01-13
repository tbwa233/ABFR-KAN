import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np

"""In the line below, make sure the .csv file corresponds with the ABIDE site you want to train on.
We provide .csv files for the NYU and UM sites, but other sites can be used so long as you provide a .csv file for them."""
data_fold = pd.read_csv(f'./UM110.csv', skip_blank_lines=True)

for index, row in data_fold.iterrows():

    name = row['SUB_ID']
    label = row['DX_GROUP']-1
    FCMatrix_path = glob(f'/path/to/Data_Preparation/1_GridBasedAnchorSelection_RandomPatchSampling/Result_FCandSignal_BasedPatch_Anchor/UM_PatchSize8_112AnchorNum/FCMatrix/UM_1_00{name}_func_preproc.mat')
    Posi_Signal_path = glob(f'/path/to/Data_Preparation/1_GridBasedAnchorSelection_RandomPatchSampling/Result_FCandSignal_BasedPatch_Anchor/UM_PatchSize8_112AnchorNum/Position_and_ROISignals/UM_1_00{name}_func_preproc.mat')

    FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix']
    Posi_Signal = loadmat(Posi_Signal_path[0])['Position_and_ROISignals']

    print(name, label, end=' ')
    print(Posi_Signal.shape)

    FCMatrixfinal = FCMatrix[:256, 256:]
    print('FCMatrixfinal_shape: ', FCMatrixfinal.shape)

    posi = Posi_Signal[:, :3]

    os.makedirs('subjectwise_position_grid_random', exist_ok=True)
    np.save(os.path.join('subjectwise_position_grid_random', f'{name}.npy'), posi)
    os.makedirs(f'./FCmatrix_grid_random/{name}', exist_ok=True)
    np.save(os.path.join('./FCmatrix_grid_random', str(name), f'fcmatrix.npy'), FCMatrixfinal)
