import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob

imgs = [nib.load(f"../Data/processed_data/train_images/BraTS-GLI-00006-101_t1c.nii").get_fdata().astype(np.float32)[:, :, 75],
        nib.load(f"../Data/processed_data/train_images/BraTS-GLI-00006-101_t1n.nii").get_fdata().astype(np.float32)[:, :, 75],
        nib.load(f"../Data/processed_data/train_images/BraTS-GLI-00006-101_t2w.nii").get_fdata().astype(np.float32)[:, :, 75],
        nib.load(f"../Data/processed_data/train_images/BraTS-GLI-00006-101_t2f.nii").get_fdata().astype(np.float32)[:, :, 75],]
lbl = nib.load("../Data/processed_data/train_masks/BraTS-GLI-00006-101-seg.nii").get_fdata().astype(np.uint8)[:, :, 75]
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
for i, img in enumerate(imgs):
    ax[i].imshow(img, cmap='gray')
    ax[i].axis('off')
ax[-1].imshow(lbl, vmin=0, vmax=4)
ax[-1].axis('off')
plt.tight_layout()
plt.show()

print("Task Finished.")
