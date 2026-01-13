import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np


data_fold = pd.read_csv(f'./UM110.csv', skip_blank_lines=True)

for index, row in data_fold.iterrows():

    name = row['SUB_ID']
    label = row['DX_GROUP']-1
    FCMatrix_path = glob(f'/localdisk0/ABFR-KAN/Data_Preparation/2_RandomAnchorSelection_RandomPatchSampling/Result_FCandSignal_BasedPatch_Anchor/UM_PatchSize8_112AnchorNum/FCMatrix/UM_1_00{name}_func_preproc.mat')
    Posi_Signal_path = glob(f'/localdisk0/ABFR-KAN/Data_Preparation/2_RandomAnchorSelection_RandomPatchSampling/Result_FCandSignal_BasedPatch_Anchor/UM_PatchSize8_112AnchorNum/Position_and_ROISignals/UM_1_00{name}_func_preproc.mat')

    FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix']
    Posi_Signal = loadmat(Posi_Signal_path[0])['Position_and_ROISignals']

    print(name, label, end=' ')
    print(Posi_Signal.shape)

    FCMatrixfinal = FCMatrix[:256, 256:]
    print('FCMatrixfinal_shape: ', FCMatrixfinal.shape)

    posi = Posi_Signal[:, :3]

    os.makedirs('subjectwise_position_random_random', exist_ok=True)
    np.save(os.path.join('subjectwise_position_random_random', f'{name}.npy'), posi)
    os.makedirs(f'./FCmatrix_random_random/{name}', exist_ok=True)
    np.save(os.path.join('./FCmatrix_random_random', str(name), f'fcmatrix.npy'), FCMatrixfinal)
