import cv2
import os
import pdb
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelBinarizer
import SimpleITK as sitk

import matplotlib.pyplot as plt

Images_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/imagesTr'
Labels_root='/home/ubuntu/DL_Project/nnunetv2/Brats-2024/nnUNet_raw/Dataset137_BraTS2024/labelsTr'


#------------------------------- image view -----------------------------------
# images = os.listdir(Images_root)
# image_path=os.path.join(Images_root,images[0])
# image = nib.load(image_path)
# data = image.get_fdata()
#
#
# print(data.shape)
# slice_index = data.shape[2] // 2  # Mid-slice for example
#
# plt.imshow(data[50,:,:], cmap='gray')
# plt.title('MRI Slice')
# plt.axis('off')  # Turn off axis numbers and ticks
# plt.show()

#--------------------------- Normalisation ----------------------------

# def load_nifti_file(file_path):
#     return nib.load(file_path).get_fdata()
#
# # Example usage: Load a single image and its label
# image_files = sorted(os.listdir(Images_root))
# label_files = sorted(os.listdir(Labels_root))
#
# sample_image = load_nifti_file(os.path.join(Images_root, image_files[0]))
# sample_label = load_nifti_file(os.path.join(Labels_root, label_files[0]))
#
# def normalize_image(image):
#     return (image - np.mean(image)) / np.std(image)
#
# normalized_image = normalize_image(sample_image)
#
# def plot_slice(img, title="MRI Slice"):
#     plt.imshow(img, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
#
# # Plot a middle slice
# slice_idx = sample_image.shape[2] // 2  # Assuming slicing along the third dimension
# # plot_slice(sample_image[90, :, :], "regular image Slice")
# # plot_slice(normalized_image[90, :, :], "Normalized MRI Slice")
#
#
# #-----------------------------comparison  of regular and normalised image -----------------------
# def plot_comparison(img1, img2, slice_idx, cmin, cmax):
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].imshow(img1[slice_idx, :, :], cmap='gray', vmin=cmin, vmax=cmax)
#     ax[0].set_title("Regular Image Slice")
#     ax[0].axis('off')
#
#     ax[1].imshow(img2[slice_idx, :, :], cmap='gray', vmin=cmin, vmax=cmax)
#     ax[1].set_title("Normalized MRI Slice")
#     ax[1].axis('off')
#
#     plt.show()
#
# # Determine common display range based on the original image
# common_min = sample_image.min()
# common_max = sample_image.max()
# print(common_min, common_max)
#
# print(normalized_image.min(), normalized_image.max())
#
# # Display using common intensity range
# # plot_comparison(sample_image, normalized_image, 90, common_min, common_max)
#
# #-------------------------------- visualising after normalisation ---------------------------
#
# def plot_normalized_image(image, slice_idx, vmin, vmax):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(image[slice_idx, :, :], cmap='gray', vmin=vmin, vmax=vmax)
#     plt.title("Normalized MRI Slice")
#     plt.axis('off')
#     plt.show()
#
# # Using the min and max values from your normalized image
# # plot_normalized_image(normalized_image, 90, -0.439, 5.996)
#
# #----------------------------Bias Field Correction ------------------------
# def correct_bias_field(image_path):
#     input_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
#     corrector = sitk.N4BiasFieldCorrectionImageFilter()
#     output_image = corrector.Execute(input_image)
#     return sitk.GetArrayFromImage(output_image)
#
#
# # sample_image = os.path.join(Images_root, image_files[0])
# # corrected_image = correct_bias_field(sample_image)
# # normalized_corrected_image = normalize_image(corrected_image)
# # plot_slice(normalized_corrected_image[:, :, normalized_corrected_image.shape[2] // 2])
# #
# # plot_slice(normalized_corrected_image[:, :, corrected_image.shape[2] // 2])
#
#
#
# #--------------------------------
