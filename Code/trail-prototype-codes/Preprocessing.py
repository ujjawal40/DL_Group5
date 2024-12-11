import cv2
import os
import pdb
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelBinarizer
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


Images_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/imagesTr'
Labels_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/labelsTr'

def normalize_data(data):
    # Applying Z-score normalization
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def correct_bias_field(image_path):
    input_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_image = corrector.Execute(input_image)
    return sitk.GetArrayFromImage(output_image)


def resample_image(data, original_affine, new_spacing=[1.0, 1.0, 1.0]):
    # Calculate the current spacing from the affine matrix
    original_spacing = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))

    # Calculate the zoom factors for the resampling
    zoom_factors = original_spacing / new_spacing

    # Resample the data using scipy's zoom function
    resampled_data = zoom(data, zoom_factors, order=1)  # Using linear interpolation

    # Update the affine matrix to reflect the new spacing
    new_affine = original_affine.copy()
    new_affine[:3, :3] = original_affine[:3, :3] / zoom_factors[:, np.newaxis]

    return resampled_data, new_affine
