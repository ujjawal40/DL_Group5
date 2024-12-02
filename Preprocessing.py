import cv2
import os
import pdb
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelBinarizer
import SimpleITK as sitk


Images_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/imagesTr'
Labels_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/labelsTr'

def normalize_image(image):
    return (image - np.mean(image)) / np.std(image)

def correct_bias_field(image_path):
    input_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output_image = corrector.Execute(input_image)
    return sitk.GetArrayFromImage(output_image)


def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    # Get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate new size using original spacing and new spacing
    new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]
    resample = sitk.ResampleImageFilter()

    # Set resampling parameters
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())

    # Apply resampling
    resampled_image = resample.Execute(image)
    return resampled_image