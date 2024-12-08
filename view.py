import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Paths
# script_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.join(script_dir, '../Data')
# train_images_dir = os.path.join(base_dir, 'training_data1_v2/train_images')
# train_masks_dir = os.path.join(base_dir, 'training_data1_v2/train_masks')

script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the Code folder
dl_group5_dir = os.path.dirname(script_dir)  # Path to DL_Group5 folder
data_dir = os.path.join(dl_group5_dir, 'nData')  # Correct path to Data folder
# data_dir = os.path.join(base_data_dir, 'training_data1_v2')  # Path to training_data1_v2
train_images_dir = os.path.join(data_dir, 'train_images')
train_masks_dir = os.path.join(data_dir, 'train_masks')

# Load images
imgs = [
    nib.load(os.path.join(train_images_dir, "BraTS-GLI-02665-100_t1c.nii.gz")).get_fdata().astype(np.float32)[:, :, 75],
    nib.load(os.path.join(train_images_dir, "BraTS-GLI-02665-100_t1n.nii.gz")).get_fdata().astype(np.float32)[:, :, 75],
    nib.load(os.path.join(train_images_dir, "BraTS-GLI-02665-100_t2w.nii.gz")).get_fdata().astype(np.float32)[:, :, 75],
    nib.load(os.path.join(train_images_dir, "BraTS-GLI-02665-100_t2f.nii.gz")).get_fdata().astype(np.float32)[:, :, 75],
]
lbl = nib.load(os.path.join(train_masks_dir, "BraTS-GLI-02665-100_seg.nii.gz")).get_fdata().astype(np.uint8)[:, :, 75]

# Plot images
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 10))
for i, img in enumerate(imgs):
    ax[i].imshow(img, cmap='gray')
    ax[i].axis('off')
ax[-1].imshow(lbl, cmap='nipy_spectral', vmin=0, vmax=4)
ax[-1].axis('off')
plt.tight_layout()
plt.show()

print("Task Finished.")

