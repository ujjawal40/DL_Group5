import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob

imgs = [nib.load(f"../Data/training_data1_v2/BraTS-GLI-02194-104/BraTS-GLI-02194-104-{m}.nii").get_fdata().astype(np.float32)[:, :, 75] for m in ["t1c", "t1n", "t2f", "t2w"]]
lbl = nib.load("../Data/training_data1_v2/BraTS-GLI-02194-104/BraTS-GLI-02194-104-seg.nii").get_fdata().astype(np.uint8)[:, :, 75]
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
for i, img in enumerate(imgs):
    ax[i].imshow(img, cmap='gray')
    ax[i].axis('off')
ax[-1].imshow(lbl, vmin=0, vmax=4)
ax[-1].axis('off')
plt.tight_layout()
plt.show()



# BraTS-GLI-02194-104