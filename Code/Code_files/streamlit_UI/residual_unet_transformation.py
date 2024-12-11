import streamlit as st
import os
import random
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
# from model_3dRESN_unet import ResidualUNet3D, Down, Up, OutConv
from residual_model import *
from unet_model import *
from scipy.spatial.distance import directed_hausdorff
from config import config_paths as configuration




def resize_image(image, new_size):
    """Resize images to a new size using SimpleITK."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * (image.GetSize()[i] / new_size[i]) for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.AffineTransform(3))
    return resampler.Execute(image)

def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)
def normalize_image(image_np):
    """Normalize image array to have pixel values between 0 and 1."""
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np

def apply_transformations(image, transform_type):
    """Apply transformations such as flip, rotation, or scale to the image."""
    if transform_type == "Flip":
        return sitk.Flip(image, [True, False, False])
    elif transform_type == "Rotation":
        transform = sitk.AffineTransform(3)
        transform.Rotate(1, 2, np.pi / 4)  # Example: rotation about the axis
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        return resampler.Execute(image)
    elif transform_type == "Scale":
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([s / 1.5 for s in image.GetSpacing()])
        resampler.SetSize([int(sz * 1.5) for sz in image.GetSize()])
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(image)
    return image  # No transformation if "None"


def residual_model(images, segmentations):
    model = ResidualUNet3D(n_channels=4, n_classes=5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Update the path to the ResidualUNet3D checkpoint you saved during training
    model_path=configuration.residual_model_path
    state_dict = torch.load(model_path,
                            map_location=device)


    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    num_classes = 5
    images = images.to(device)
    images = images.to(device).unsqueeze(0) if images.dim() == 4 else images.to(device)
    segmentations = segmentations
    segmentations_dev = segmentations.to(device)
    images = images
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)
    dice_score = dice_coefficient(outputs, segmentations_dev, 5)
    fig=visualize_predictions(images.cpu().numpy(), segmentations.numpy(), predictions.cpu().numpy())
    return fig,dice_score
def dice_coefficient(predictions, targets, num_classes):
    """Calculate the Dice coefficient for each class."""
    dice_scores = []
    predictions = torch.argmax(predictions, dim=1)
    for c in range(1, num_classes):  # Skip class 0 (background)
        pred_c = (predictions == c).float()
        target_c = (targets == c).float()
        intersection = torch.sum(pred_c * target_c)
        dice_scores.append((2. * intersection) / (torch.sum(pred_c) + torch.sum(target_c)))
    return dice_scores

def create_custom_colormap():
    colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    cmap_name = 'tumor_segmentation'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

# Visualization adapted for Streamlit
def visualize_predictions(images, targets, predictions, slice_idx=None):
    print("Images Shape:", images.shape)  # Expected to be [1, Channels, Depth, Height, Width]
    print("Segmentations Shape:", targets.shape)  # Expected to be [1, Depth, Height, Width]
    print("Predictions Shape:", predictions.shape)
    cmap = create_custom_colormap()

    # Ensure we are squeezing only if the batch dimension is indeed 1 to avoid removing other dimensions
    if images.shape[0] == 1:
        images = images.squeeze(0)
    if predictions.shape[0] == 1:
        predictions = predictions.squeeze(0)

    # Make sure the slice index is within bounds
    if slice_idx is None or slice_idx >= images.shape[1]:  # Check if the slice_idx is within the depth dimension
        slice_idx = images.shape[1] // 2  # Safe default slice index

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Adjust indexing based on the actual dimension sizes
    input_image = images[0, slice_idx, :, :]  # Assuming channel first after batch
    target_mask = targets[slice_idx, :, :]  # Assuming single channel mask
    predicted_mask = predictions[slice_idx, :, :]  # Assuming single channel prediction
    # dice_score = dice_coefficient(target_mask, predicted_mask)
    # hausdorff_distance = hausdorff_95(target_mask, predicted_mask)
    # # print("Dice Score:", dice_score)
    # # print("hausdorff_distance",hausdorff_distance)
    # print("target mask",target_mask.shape)
    # print("predict mask", predicted_mask.shape)

    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input Image (Slice)')
    axes[1].imshow(target_mask, cmap=cmap)
    axes[1].set_title('Ground Truth (Slice)')
    axes[2].imshow(predicted_mask, cmap=cmap)
    axes[2].set_title('Prediction (Slice)')
    plt.tight_layout()

    return fig

def predicting_tab_residual_with_transformation():
    st.title("3D U-Net with Residual Blocks  Prediction with Image Transformation")
    path = configuration.val_images_dir
    mask_path = configuration.val_mask_dir
    images_unet = os.listdir(path)

    image_selection_unet = st.selectbox("Select an NII file for prediction with 3D UNET after Transformation", options=images_unet, key="res_trans")

    # Select the type of transformation
    transform_options = ["None", "Flip", "Rotation", "Scale"]
    selected_transformation = st.selectbox("Select the type of transformation to apply", options=transform_options, key="transformselection")

    if st.button("Apply Transformation and Predict Resunet"):
        patient_id = image_selection_unet.split('_')[0]
        modalities = ['t1c', 't1n', 't2f', 't2w']
        images_original = []
        for mod in modalities:
            mod_path = os.path.join(path, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)
            # Apply transformation to each modality
            transformed_image = apply_transformations(image, selected_transformation)
            transformed_image = resize_image(transformed_image, (132, 132, 116))
            transformed_image_np = sitk.GetArrayFromImage(transformed_image)
            normalized_transformed = normalize_image(transformed_image_np)
            images_original.append(normalized_transformed)

        image_stack_transformed = np.stack(images_original, axis=0)
        images_tensor = torch.tensor(image_stack_transformed, dtype=torch.float32)

        seg_path = os.path.join(mask_path, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)
        # Apply the same transformation to the segmentation
        transformed_segmentation = apply_transformations(segmentation, selected_transformation)
        transformed_segmentation = resize_image(transformed_segmentation, (132, 132, 116))
        transformed_seg_array = sitk.GetArrayFromImage(transformed_segmentation)
        masks_tensor = torch.tensor(transformed_seg_array, dtype=torch.long)

        fig, dice_score = residual_model(images_tensor, masks_tensor)
        st.pyplot(fig)
        st.subheader("Results")
        for i, score in enumerate(dice_score):
            st.write(f"*Dice Coefficient* class {i+1}: {score:.4f}")
