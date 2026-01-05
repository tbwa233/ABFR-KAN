# I wrote this
import os
import SimpleITK as sitk
from scipy import io
import numpy as np


def calc_mean_matrix(fmri_path, GMmask_path, anchor_arr_3d, patch_size):
    fmri_arr = sitk.GetArrayFromImage(sitk.ReadImage(fmri_path))
    GMmask_arr_3d = sitk.GetArrayFromImage(sitk.ReadImage(GMmask_path))

    print('fmri_arr.shape:', fmri_arr.shape)
    print('GMmask_arr_3d.shape:', GMmask_arr_3d.shape)
    print('anchor_arr_3d.shape:', anchor_arr_3d.shape)

    fmri_arr = np.nan_to_num(fmri_arr)
    GMmask_arr_4d = np.expand_dims(GMmask_arr_3d, axis=0).repeat(fmri_arr.shape[0], axis=0)
    anchor_arr_4d = np.expand_dims(anchor_arr_3d, axis=0).repeat(fmri_arr.shape[0], axis=0)

    print('GMmask_arr_4d.shape:', GMmask_arr_4d.shape)
    print('anchor_arr_4d.shape:', anchor_arr_4d.shape)

    half_size = patch_size // 2
    x_bag = np.random.choice(61, 10000, replace=True)
    y_bag = np.random.choice(73, 10000, replace=True)
    z_bag = np.random.choice(61, 10000, replace=True)

    stopflag = 0
    result = []
    position_add_result = []
    for k in range(10000):
        whole = np.zeros((61, 73, 61))  # zyx
        x_index = x_bag[k]
        y_index = y_bag[k]
        z_index = z_bag[k]
        whole[max(0, (x_index - half_size)):min((x_index + half_size), 61),
              max(0, (y_index - half_size)):min((y_index + half_size), 73),
              max(0, (z_index - half_size)):min((z_index + half_size), 61)] = 1
        whole = np.expand_dims(whole, axis=0).repeat(fmri_arr.shape[0], axis=0)

        patch_remove = np.multiply(GMmask_arr_4d, whole)
        if np.sum(patch_remove) < 1:
            print('drop')
            continue
        else:
            stopflag += 1
            print('patch:', k)
            result_part = np.multiply(fmri_arr, patch_remove)
            sum_val = np.sum(patch_remove, axis=(1, 2, 3))[0]
            result_mean = np.sum(result_part, axis=(1, 2, 3)) / sum_val

            position = [x_index, y_index, z_index] / np.array([61.0, 73.0, 61.0])
            position_add_result_mean = np.hstack((position, result_mean))

            print('position_add_result_mean.shape:', position_add_result_mean.shape)

            result.append(result_mean)
            position_add_result.append(position_add_result_mean)
        if stopflag == 256:
            break

    anchorresult = []
    for i in range(1, 113):  # (1, anchorNum + 1)
        anchor_part = np.where(anchor_arr_4d == i, 1, 0)
        anchorresult_part = np.multiply(fmri_arr, np.multiply(GMmask_arr_4d, anchor_part))
        sum_val = np.sum(anchor_part, axis=(1, 2, 3))[0]
        anchorresult_mean = np.sum(anchorresult_part, axis=(1, 2, 3)) / sum_val
        anchorresult.append(anchorresult_mean)

    anchorresult = np.asarray(anchorresult, np.float64)
    result = np.asarray(result, np.float64)
    position_add_result = np.asarray(position_add_result)

    cc_matrix = np.corrcoef(result, anchorresult)
    print('cc_matrix.shape:', cc_matrix.shape)
    return np.nan_to_num(cc_matrix), position_add_result

num_iterations = 3  # Number of sampling iterations
patch_size_variations = [8, 12, 16]  # Vary the patch sizes. These can be any value (within reason, obviously)

result_dir = './Result_FCandSignal_BasedPatch_Anchor/NewUM_PatchSize_Variations'
image_dir = '/localdisk1/Datasets/ABIDE/Sites/UM'
mask_dir = '/localdisk0/ABFR-KAN/Data_Preparation/template'
name_list = os.listdir(f'{image_dir}')
name_list.sort()

anchor_arr_3 = sitk.GetArrayFromImage(
    sitk.ReadImage('./AnchorMask/FixedCoordinateWithGMmask_forAnchorSize32/AnchorPatch_mask_112AnchorNum_617361.nii')
)

print('anchor_arr_3.shape:', anchor_arr_3.shape)

# Iterative sampling and processing
for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1} of {num_iterations}")

    # Dynamically adjust patch size for each iteration
    current_patch_size = patch_size_variations[iteration % len(patch_size_variations)]
    iteration_result_dir = os.path.join(result_dir, f'Iteration_{iteration + 1}')
    os.makedirs(iteration_result_dir, exist_ok=True)

    for name in name_list:
        print(name, end='  ')
        fmri_path = os.path.join(f'{image_dir}', f'{name}')
        GMmask_path = os.path.join(f'{mask_dir}', 'GreyMatterMask_617361.nii')

        cc_matrix, result = calc_mean_matrix(fmri_path, GMmask_path, anchor_arr_3, current_patch_size)

        print('Result for', name, 'cc_matrix.shape:', cc_matrix.shape, 'result.shape:', result.shape)

        # Save FC matrices
        fc_path = os.path.join(f'{iteration_result_dir}/FCMatrix')
        os.makedirs(fc_path, exist_ok=True)
        io.savemat(os.path.join(fc_path, f'{name[:-7]}.mat'), {'cc_matrix': cc_matrix})

        # Save position and ROI signals
        signal_path = os.path.join(f'{iteration_result_dir}/Position_and_ROISignals')
        os.makedirs(signal_path, exist_ok=True)
        io.savemat(os.path.join(signal_path, f'{name[:-7]}.mat'), {'Position_and_ROISignals': result})

# Combine results across iterations
combined_cc_matrices = []
combined_position_results = []

for iteration in range(num_iterations):
    iteration_result_dir = os.path.join(result_dir, f'Iteration_{iteration + 1}')
    fc_path = os.path.join(f'{iteration_result_dir}/FCMatrix')
    signal_path = os.path.join(f'{iteration_result_dir}/Position_and_ROISignals')

    for name in name_list:
        fc_matrix = io.loadmat(os.path.join(fc_path, f'{name[:-7]}.mat'))['cc_matrix']
        signal_result = io.loadmat(os.path.join(signal_path, f'{name[:-7]}.mat'))['Position_and_ROISignals']

        combined_cc_matrices.append(fc_matrix)
        combined_position_results.append(signal_result)

# Aggregate results
aggregated_fc_matrix = np.mean(combined_cc_matrices, axis=0)  # Average FC matrices
aggregated_position_results = np.vstack(combined_position_results)  # Concatenate position results

print('aggregated_fc_matrix.shape:', aggregated_fc_matrix.shape)
print('aggregated_position_results.shape:', aggregated_position_results.shape)

# Save aggregated results
final_result_dir = os.path.join(result_dir, 'FinalAggregatedResults')
os.makedirs(final_result_dir, exist_ok=True)
io.savemat(os.path.join(final_result_dir, 'Aggregated_FCMatrix.mat'), {'aggregated_fc_matrix': aggregated_fc_matrix})
io.savemat(os.path.join(final_result_dir, 'Aggregated_Position_Results.mat'),
           {'aggregated_position_results': aggregated_position_results})
