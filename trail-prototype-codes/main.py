import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from Preprocessing import normalize_data, resample_image
import os
def load_image(path):
    return nib.load(path)

def save_image(data, affine, save_path):
    """Save a NIfTI image using nibabel."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, save_path)



Images_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/imagesTr'
Labels_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/labelsTr'
save_dir='/home/ubuntu/Data-prrocessing/preprocessed'


images=os.listdir(Images_root)
for image_name in images:
    image_path=os.path.join(Images_root,image_name)
    image=load_image(image_path)
    data = image.get_fdata()
    normalized_data = normalize_data(data)
    save_path = os.path.join(save_dir, image_name)
    if not save_path.endswith('.nii'):
        save_path += '.nii'  # Ensure the file has a .nii extension

    # Save the normalized image
    save_image(normalized_data, image.affine, save_path)