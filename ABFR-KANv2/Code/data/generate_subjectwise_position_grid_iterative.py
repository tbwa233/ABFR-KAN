import os
from glob import glob
import pandas as pd
from scipy.io import loadmat
import numpy as np

base_dir = '/localdisk0/ABFR-KAN/Data_Preparation/3_GridBasedAnchorSelection_IterativePatchSampling/Result_FCandSignal_BasedPatch_Anchor/NewUM_PatchSize_Variations'
iterations = [f'Iteration_{i}' for i in range(1, 4)]
data_fold = pd.read_csv(f'./UM110.csv', skip_blank_lines=True)

os.makedirs('subjectwise_position_grid_iterative', exist_ok=True)
os.makedirs('FCmatrix_grid_iterative', exist_ok=True)

for index, row in data_fold.iterrows():

    name = row['SUB_ID']
    label = row['DX_GROUP'] - 1
    print(f"Processing subject {name} with label {label}")

    all_FCMatrices = []
    all_PositionSignals = []

    for iteration in iterations:
        FCMatrix_path = glob(
            os.path.join(base_dir, iteration, 'FCMatrix', f'UM_1_00{name}_func_preproc.mat')
        )
        Posi_Signal_path = glob(
            os.path.join(base_dir, iteration, 'Position_and_ROISignals', f'UM_1_00{name}_func_preproc.mat')
        )

        if not FCMatrix_path or not Posi_Signal_path:
            print(f"Missing data for subject {name} in {iteration}")
            continue

        FCMatrix = loadmat(FCMatrix_path[0])['cc_matrix']
        Posi_Signal = loadmat(Posi_Signal_path[0])['Position_and_ROISignals']

        print(f"{iteration} - FCMatrix shape: {FCMatrix.shape}, Position Signal shape: {Posi_Signal.shape}")

        all_FCMatrices.append(FCMatrix)
        all_PositionSignals.append(Posi_Signal)

    if all_FCMatrices:
        combined_FCMatrix = np.mean(all_FCMatrices, axis=0)
        print(f"Combined FCMatrix shape: {combined_FCMatrix.shape}")

        combined_PositionSignals = np.vstack(all_PositionSignals)
        print(f"Combined Position Signals shape: {combined_PositionSignals.shape}")

        FCMatrixfinal = combined_FCMatrix[:256, 256:]
        posi = combined_PositionSignals[:, :3]

        np.save(os.path.join('subjectwise_position_grid_iterative', f'{name}.npy'), posi)

        subject_fc_dir = os.path.join('./FCmatrix_grid_iterative', str(name))
        os.makedirs(subject_fc_dir, exist_ok=True)
        np.save(os.path.join(subject_fc_dir, 'fcmatrix.npy'), FCMatrixfinal)

    else:
        print(f"No data found for subject {name} across all iterations.")
