import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import random

# Load the gray matter mask
GMmask = sitk.GetArrayFromImage(sitk.ReadImage('/localdisk0/ABFR-KAN/Data_Preparation/template/GreyMatterMask_181217181.nii.gz'))

# Initialize variables
wholefinal = np.zeros((181, 217, 181))
label = 1
patch_size = 32  # Anchor size
half_size = patch_size // 2
label_index = []

# Randomly sample anchor patches
for i in range(150):  # Number of desired random anchor patches
    while True:  # Ensure valid random patch
        x_start = random.randint(0, 181 - patch_size)
        y_start = random.randint(0, 217 - patch_size)
        z_start = random.randint(0, 181 - patch_size)
        
        whole = np.zeros((181, 217, 181))
        whole[x_start:x_start + patch_size, y_start:y_start + patch_size, z_start:z_start + patch_size] = 1
        
        patch_remove = np.multiply(whole, GMmask)
        if np.sum(patch_remove) >= 100:  # Ensure sufficient overlap with gray matter
            break

    # Calculate the center coordinates of the patch
    x_index = x_start + half_size
    y_index = y_start + half_size
    z_index = z_start + half_size
    label_index.append([x_index, y_index, z_index, label, np.sum(patch_remove)])
    wholefinal[x_start:x_start + patch_size, y_start:y_start + patch_size, z_start:z_start + patch_size] = label

    label += 1

print('AnchorNum: ', label)

# Save anchor patch data
label_index_array = np.array(label_index)
anchor_path = './AnchorMask/RandomAnchorNewFixedCoordinateWithGMmask_forAnchorSize32'
os.makedirs(f"{anchor_path}", exist_ok=True)
np.save(f'{anchor_path}/AnchorPatch_index.npy', label_index_array)
np.save(f'{anchor_path}/AnchorPatch_mask.npy', wholefinal)

df = pd.DataFrame(label_index_array)
df.to_csv(f'{anchor_path}/AnchorPatch_index.csv')

# Save the anchor patch mask as a .nii file
origin_image = sitk.ReadImage('/localdisk0/ABFR-KAN/Data_Preparation/template/ch2bet.nii')
whole_image = sitk.GetImageFromArray(wholefinal)
whole_image.SetDirection(origin_image.GetDirection())
whole_image.SetOrigin(origin_image.GetOrigin())
whole_image.SetSpacing(origin_image.GetSpacing())

sitk.WriteImage(whole_image, f'{anchor_path}/AnchorPatch_mask_181217181.nii')

# Reslice the anchor patch mask to match a reference volume
volume = sitk.ReadImage(f'{anchor_path}/AnchorPatch_mask_181217181.nii')
reference_volume = sitk.ReadImage(f'/localdisk0/ABFR-KAN/Data_Preparation/template/BrainMask_05_617361.nii')
resliced_mask = sitk.Resample(volume, referenceImage=reference_volume,
                              transform=sitk.Transform(),
                              interpolator=sitk.sitkNearestNeighbor,
                              defaultPixelValue=0.0)

sitk.WriteImage(resliced_mask, f'{anchor_path}/AnchorPatch_mask_{label-1}AnchorNum_617361.nii')
