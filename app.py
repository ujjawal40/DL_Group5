import streamlit as st
import os
from config import config_paths as configuration
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
from unet_predicting_transformation import *
from residual_unet_transformation import *
from scipy.spatial.distance import directed_hausdorff



# Title and Header

# Inject CSS for styling
# Inject CSS for styling

def Home():
    st.title("Brain Tumor Segmentation")
    st.subheader("Team Members:")
    st.write("- Ujjawal")
    st.write("- Bala Krishna")
    st.write("- Bhagawath Sai")
    st.write("- Sai Avinash")

    st.header("Project Relevance")
    st.write("""
    Brain tumor segmentation is a crucial task in medical imaging that assists in the diagnosis and treatment of patients. 
    Accurate segmentation helps radiologists and clinicians understand tumor regions, identify malignancies, and monitor treatment progress.
    The project aims to implement deep learnoing models (3D FCNNs, Unet, Residual Unet) to segment various regions of brain tumors from multi-modal MRI scans.
    """)

    st.header("Dataset Description")
    st.write("""
    This project utilizes the BraTS dataset, which contains pre-processed multi-modal MRI scans for brain tumor segmentation.
    The dataset includes the following modalities:
    - T1-weighted (t1n)
    - T1-weighted post-contrast (t1c)
    - T2-weighted (t2w)
    - Fluid-attenuated inversion recovery (t2f)
    """)

    st.header("Types of Scans and Tumor Regions")
    st.write("""
    The dataset includes annotations for various tumor regions:
    - *Whole Tumor (WT)*: All visible tumor regions.
    - *Necrotic and Non-enhancing Tumor Core (NETC)*: Central dead tissue and non-enhancing regions.
    - *Enhancing Tumor (ET)*: Actively growing tumor cells.
    - *Surrounding Non-tumor Fluid Heterogeneity (SNFH)*: Swelling or fluid accumulation around the tumor.
    - *Resection Cavity*: Post-surgical void in the brain.
    """)

    # Tumor regions selection
    tumor_regions = {
        "Whole Tumor (WT)": "All visible tumor regions.",
        "Necrotic and Non-enhancing Tumor Core (NETC)": "Central dead tissue and non-enhancing regions.",
        "Enhancing Tumor (ET)": "Actively growing tumor cells.",
        "Surrounding Non-tumor Fluid Heterogeneity (SNFH)": "Swelling or fluid accumulation around the tumor.",
        "Resection Cavity": "Post-surgical void in the brain."
    }

    selected_region = st.radio("Select Tumor Region:", list(tumor_regions.keys()))
    st.write(f"*Description of Selected Region:* {tumor_regions[selected_region]}")



def load_image_shape(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)


def image_shape():
    subjects_dir = configuration.brats_2024_dir
    subjects = os.listdir(subjects_dir)

    patient_dir = random.choice(subjects)

    patient_dir_path = os.path.join(subjects_dir, patient_dir)
    patient_mri_scans = os.listdir(patient_dir_path)

    image_path=os.path.join(patient_dir_path,patient_mri_scans[0])
    image = load_image(image_path)
    image_np = sitk.GetArrayFromImage(image)

    # Print the shape of the image
    print("Shape of the loaded image:", image_np.shape)
    st.write("Shape of the Raw image:", image_np.shape)



def load_mri_data(patient_dir_path, selected_category):
    for file_name in os.listdir(patient_dir_path):
        if file_name.startswith(selected_category) and file_name.endswith('.nii'):
            file_path = os.path.join(patient_dir_path, file_name)
            mri_data = nib.load(file_path)
            return mri_data.get_fdata()
    return None
def visualization():

    subjects_dir = configuration.brats_2024_dir
    subjects = os.listdir(subjects_dir)

    patient_dir = random.choice(subjects)

    patient_dir_path=os.path.join(subjects_dir, patient_dir)
    patient_mri_scans = os.listdir(patient_dir_path)



    st.header("Image Viewer")

    categories = ['t1c','t1n','t2f','t2w','seg']
    view_categories = ['Axial', 'Sagittal', 'Coronal']

    # Streamlit UI for view category selection; default is 'Axial'
    selected_view_category = st.selectbox("Select a view category", options=view_categories,
                                          index=view_categories.index('Axial'))

    # Display MRI scans
    # Using columns to display images category-wise
    cols = st.columns(len(categories))
    for idx, category in enumerate(categories):
        with cols[idx]:
            st.subheader(f"Category: {category}")
            files_displayed = 0  # Counter to check if any files are displayed
            for file in os.listdir(patient_dir_path):
                pattern=f'-{category}.nii.gz'
                if file.endswith(pattern) and file.endswith('.nii.gz'):
                    file_path = os.path.join(patient_dir_path, file)
                    mri_data = nib.load(file_path).get_fdata()

                    if selected_view_category == 'Axial':
                        slice_selected = mri_data.shape[2] // 2
                        image = mri_data[:, :, slice_selected]
                    elif selected_view_category == 'Sagittal':
                        slice_selected = mri_data.shape[0] // 2
                        image = mri_data[slice_selected, :, :]
                    elif selected_view_category == 'Coronal':
                        slice_selected = mri_data.shape[1] // 2
                        image = mri_data[:, slice_selected, :]

                    # Display the image using matplotlib
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray')
                    ax.axis('off')  # Hide axes
                    st.pyplot(fig)
                    files_displayed += 1

            if files_displayed == 0:
                st.write(f"No MRI data available for the category: {category}")

#--------------------------------------
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#
#         # Skip connection: if input channels and output channels are different, use a 1x1 convolution to match dimensions
#         self.skip_connection = nn.Conv3d(in_channels, out_channels,
#                                          kernel_size=1) if in_channels != out_channels else None
#
#     def forward(self, x):
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         # Skip connection
#         if self.skip_connection:
#             identity = self.skip_connection(x)
#
#         out += identity  # Add skip connection
#         out = self.relu(out)
#         return out
#
#
# class ResidualUNet3D(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(ResidualUNet3D, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = ResidualBlock(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             ResidualBlock(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# class Up(nn.Module):
#     """Upscaling then ResidualBlock"""
#
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#             self.conv = ResidualBlock(in_channels, out_channels)  # Remove in_channels // 2
#         else:
#             self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = ResidualBlock(in_channels // 2, out_channels)  # Corrected here as well
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffZ = x2.size()[2] - x1.size()[2]
#         diffY = x2.size()[3] - x1.size()[3]
#         diffX = x2.size()[4] - x1.size()[4]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2,
#                         diffZ // 2, diffZ - diffZ // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)

#---------------------------------------- Test Image  brats dataset---------------------------------
def resize_image(image, new_size):
    """Resize images to a new size using SimpleITK."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[0] * (image.GetSize()[0] / new_size[0]),
                                image.GetSpacing()[1] * (image.GetSize()[1] / new_size[1]),
                                image.GetSpacing()[2] * (image.GetSize()[2] / new_size[2])])
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


#----------------------------------------

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

#_--------------------------------------------------------------------------

def print_model_summary(model, file='model_summary_resunet.txt'):
    with open(file, 'w') as f:
        f.write(str(model))


def mask_to_coords(mask):
    """Converts a 3D binary mask to a list of coordinates."""
    coords = np.column_stack(np.nonzero(mask))
    return coords if coords.size > 0 else None

def hausdorff_distance(preds, true, num_classes):
    """Calculate the Hausdorff distance for each class."""
    preds_np = preds.squeeze().cpu().numpy()  # Assuming the prediction tensor includes class probabilities
    true_np = true.squeeze().cpu().numpy()

    class_wise_hausdorff = []

    print("Processing Hausdorff distance...")
    for c in range(1, num_classes):  # Assuming 0 is the background
        pred_mask = preds_np == c
        true_mask = true_np == c

        u = mask_to_coords(pred_mask)
        v = mask_to_coords(true_mask)

        if u is None or v is None or u.size == 0 or v.size == 0:
            class_wise_hausdorff.append(float('inf'))  # Handle empty coordinates
            print(f"Class {c}: Empty coordinates encountered.")
        else:
            print(f"Class {c} - u shape: {u.shape}, v shape: {v.shape}")
            hd1 = directed_hausdorff(u, v)[0]
            hd2 = directed_hausdorff(v, u)[0]
            class_wise_hausdorff.append(max(hd1, hd2))

    return class_wise_hausdorff



def residual_model(images, segmentations):
    model = ResidualUNet3D(n_channels=4, n_classes=5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Update the path to the ResidualUNet3D checkpoint you saved during training
    model_path=configuration.residual_model_path
    state_dict = torch.load(model_path,
                            map_location=device)


    model.load_state_dict(state_dict)
    print_model_summary(model)
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
    hausdorff_dist = hausdorff_distance(predictions, segmentations,5)
    fig=visualize_predictions(images.cpu().numpy(), segmentations.numpy(), predictions.cpu().numpy())
    return fig,dice_score,hausdorff_dist


def predicting_tab_residual():
    st.title("3D U-Net with Residual Blocks")
    path=configuration.val_images_dir
    mask_path=configuration.val_mask_dir
    images = os.listdir(path)

    image_selection=st.selectbox("Select an NII file for prediction with 3D Unet with Residual Blocks", options=images
                                          )

    if image_selection:
        patient_id = image_selection.split('_')[0]
        full_path = os.path.join(path, image_selection)
        modalities = ['t1c', 't1n', 't2f', 't2w']
        images_original = []
        for mod in modalities:
            mod_path = os.path.join(path, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)
            original_image = resize_image(image, (132, 132, 116))
            original_image_np = sitk.GetArrayFromImage(original_image)
            normalized_original = normalize_image(original_image_np)
            images_original.append(normalized_original)

        image_stack_original = np.stack(images_original, axis=0)
        print(image_stack_original.shape)
        seg_path = os.path.join(mask_path, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)
        original_segmentation = resize_image(segmentation,(132, 132, 116))
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)
        images_original=torch.tensor(image_stack_original, dtype=torch.float32)
        masks_original= torch.tensor(original_seg_array, dtype=torch.long)
        fig,dice_score,hausdorff_dist=residual_model(images_original, masks_original)
        st.pyplot(fig)
        st.subheader("Results")
        for i, score in enumerate(dice_score):
            st.write(f"*Dice Coefficient* class {i + 1}: {score:.4f}")
        for i, score in enumerate(hausdorff_dist):
            st.write(f"*hausdorff_dist score * class {i + 1}: {score:.4f}")




#------------------------------------------------------------------unet----------------------------------

def print_model_summary_unet(model, file='model_summary_3dunet.txt'):
    with open(file, 'w') as f:
        f.write(str(model))

def unet_model(images, segmentations):
    model_unet = UNet3D(n_channels=4, n_classes=5)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_unet =model_unet.to(device)
    # Update the path to the ResidualUNet3D checkpoint you saved during training
    model_path=configuration.unet_model_path
    state_dict = torch.load(model_path,
                            map_location=device)

    model_unet.load_state_dict(state_dict)

    print_model_summary_unet(model_unet)

    criterion = nn.CrossEntropyLoss()
    num_classes = 5
    images = images.to(device)
    images = images.to(device).unsqueeze(0) if images.dim() == 4 else images.to(device)
    segmentations = segmentations
    segmentations_dev = segmentations.to(device)
    images = images
    outputs = model_unet(images)
    predictions = torch.argmax(outputs, dim=1)
    print("predictions size",predictions.size())
    print("outputs size",outputs.size())
    # print("outputs size", outputs[4].size())
    dice_score=dice_coefficient(outputs,segmentations_dev,5)
    hausdorff_dist = hausdorff_distance(predictions, segmentations, 5)
    print(dice_score)
    fig=visualize_predictions(images.cpu().numpy(), segmentations.numpy(), predictions.cpu().numpy())
    return fig,dice_score,hausdorff_dist


def predicting_tab_unet():
    path=configuration.val_images_dir
    mask_path=configuration.val_mask_dir
    images_unet = os.listdir(path)

    image_selection_unet=st.selectbox("Select an NII file for prediction with 3D UNET", options=images_unet,key="unet")

    if image_selection_unet:
        patient_id = image_selection_unet.split('_')[0]
        full_path = os.path.join(path, image_selection_unet)
        modalities = ['t1c', 't1n', 't2f', 't2w']
        images_original = []
        for mod in modalities:
            mod_path = os.path.join(path, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)
            original_image = resize_image(image, (132, 132, 116))
            original_image_np = sitk.GetArrayFromImage(original_image)
            normalized_original = normalize_image(original_image_np)
            images_original.append(normalized_original)

        image_stack_original = np.stack(images_original, axis=0)
        print(image_stack_original.shape)
        seg_path = os.path.join(mask_path, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)
        original_segmentation = resize_image(segmentation,(132, 132, 116))
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)
        images_original=torch.tensor(image_stack_original, dtype=torch.float32)
        masks_original= torch.tensor(original_seg_array, dtype=torch.long)
        fig,dice_score,hausdorff_dist=unet_model(images_original, masks_original)
        st.pyplot(fig)
        st.subheader("Results")
        for i,score in enumerate(dice_score):
            st.write(f"*Dice Coefficient* class {i+1}: {score:.4f}")

        for i, score in enumerate(hausdorff_dist):
            st.write(f"*hausdorff_dist score * class {i + 1}: {score:.4f}")


def dice_coefficient(predictions, targets, num_classes):
    """
    Compute the Dice Similarity Coefficient (DSC) for each class.

    Args:
        predictions (torch.Tensor): Predicted segmentation logits [Batch Size, Classes, Depth, Height, Width].
        targets (torch.Tensor): Ground truth segmentation [Batch Size, Depth, Height, Width].
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        list: Dice coefficient for each class.
    """
    eps = 1e-5  # Small epsilon to avoid division by zero
    dice_scores = []

    # Convert logits to predicted classes
    preds = torch.argmax(predictions, dim=1)  # [Batch Size, Depth, Height, Width]

    for c in range(1, num_classes):  # Ignore class 0 (background)
        pred_c = (preds == c).float()  # Binary mask for class c in predictions
        target_c = (targets == c).float()  # Binary mask for class c in targets

        intersection = torch.sum(pred_c * target_c)
        denominator = torch.sum(pred_c) + torch.sum(target_c)

        dice_score = (2.0 * intersection + eps) / (denominator + eps)
        dice_scores.append(dice_score.item())

    return dice_scores


# def metrics
#-----------------------------------------------AUgmentation---------------------
#
#
# def load_image(file_path):
#     return sitk.ReadImage(file_path)
#
# def resize_image(image, new_size):
#     resample = sitk.ResampleImageFilter()
#     resample.SetSize(new_size)
#     resample.SetInterpolator(sitk.sitkLinear)
#     return resample.Execute(image)
#
# def normalize_image(np_image):
#     return (np_image - np.mean(np_image)) / np.std(np_image)
#
# def apply_transformations(image, transformation_type):
#     """Apply specified transformation to the image using SimpleITK."""
#     if transformation_type == "Flip":
#         transformed_image = sitk.Flip(image, [True, False, False])
#     elif transformation_type == "Rotation":
#         transform = sitk.AffineTransform(3)
#         transform.Rotate(1, 2, np.pi / 4)  # Example rotation about the axis
#         resampler = sitk.ResampleImageFilter()
#         resampler.SetReferenceImage(image)
#         resampler.SetInterpolator(sitk.sitkLinear)
#         resampler.SetTransform(transform)
#         transformed_image = resampler.Execute(image)
#     elif transformation_type == "Scale":
#         resampler = sitk.ResampleImageFilter()
#         resampler.SetOutputSpacing([s / 1.5 for s in image.GetSpacing()])
#         resampler.SetSize([int(sz * 1.5) for sz in image.GetSize()])
#         resampler.SetInterpolator(sitk.sitkLinear)
#         transformed_image = resampler.Execute(image)
#     else:
#         transformed_image = image  # No transformation if "None"
#     return transformed_image
#
# def aug_visualisation():
#     st.header("Image Augmentation Viewer")
#
#     # configuration and directory selection
#     subjects_dir = configuration.brats_2024_dir
#     subjects = os.listdir(subjects_dir)
#     patient_dir = st.selectbox("Select a directory", options=subjects, key="patient_dir_aug")
#     patient_dir_path = os.path.join(subjects_dir, patient_dir)
#
#     # View, augmentation type, and category selection
#     view_categories_aug = ['Axial', 'Sagittal', 'Coronal']
#     augment_types = ["None", "Flip", "Rotation", "Scale"]
#     selected_view_category = st.selectbox("Select a view category", options=view_categories_aug, index=0, key="view_category_aug")
#     augmentation_selection = st.selectbox("Select the type of augmentation", options=augment_types, key="augmentation_selection")
#
#     if st.button("Apply Transformation"):
#         categories = ['t1c', 't1n', 't2f', 't2w', 'seg']
#         cols = st.columns(len(categories))
#         for idx, category in enumerate(categories):
#             with cols[idx]:
#                 st.subheader(f"Category: {category}")
#                 files_displayed = 0  # Counter to check if any files are displayed
#                 for file_name in os.listdir(patient_dir_path):
#                     if category in file_name and file_name.endswith('.nii.gz'):
#                         file_path = os.path.join(patient_dir_path, file_name)
#                         sitk_image = load_image(file_path)
#
#                         # Apply transformation
#                         sitk_transformed = apply_transformations(sitk_image, augmentation_selection) if augmentation_selection != "None" else sitk_image
#                         np_image_original = sitk.GetArrayFromImage(sitk_image)
#                         np_image_transformed = sitk.GetArrayFromImage(sitk_transformed)
#
#                         # Get the slice for the selected view
#                         slice_index = {
#                             'Axial': np_image_original.shape[0] // 2,
#                             'Sagittal': np_image_original.shape[2] // 2,
#                             'Coronal': np_image_original.shape[1] // 2
#                         }[selected_view_category]
#
#                         original_image = np.take(np_image_original, slice_index, axis={'Axial': 0, 'Sagittal': 2, 'Coronal': 1}[selected_view_category])
#                         transformed_image = np.take(np_image_transformed, slice_index, axis={'Axial': 0, 'Sagittal': 2, 'Coronal': 1}[selected_view_category])
#
#                         # Create subplot for original and transformed images
#                         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#                         axes[0].imshow(original_image, cmap='gray')
#                         axes[0].set_title('Original Image')
#                         axes[0].axis('off')
#
#                         axes[1].imshow(transformed_image, cmap='gray')
#                         axes[1].set_title('Transformed Image')
#                         axes[1].axis('off')
#
#                         st.pyplot(fig)
#                         files_displayed += 1
#
#                 if files_displayed == 0:
#                     st.write(f"No MRI data available for the category: {category}")

#_---------------------------------------------------------------------------------------------------------------------------------
def load_image(file_path):
    """Load the image using SimpleITK."""
    return sitk.ReadImage(file_path)

def apply_transformations(image, transformation_type):
    """Apply specified transformation to the image using SimpleITK."""
    if transformation_type == "Flip":
        transformed_image = sitk.Flip(image, [True, False, False])
    elif transformation_type == "Rotation":
        transform = sitk.AffineTransform(3)
        transform.Rotate(1, 2, np.pi / 4)  # Example rotation about the axis
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        transformed_image = resampler.Execute(image)
    elif transformation_type == "Scale":
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([s / 1.5 for s in image.GetSpacing()])
        resampler.SetSize([int(sz * 1.5) for sz in image.GetSize()])
        resampler.SetInterpolator(sitk.sitkLinear)
        transformed_image = resampler.Execute(image)
    else:
        transformed_image = image  # No transformation if "None"
    return transformed_image

def aug_visualisation():
    st.header("Image Augmentation Viewer")

    # configuration and directory selection
    subjects_dir = configuration.brats_2024_dir
    subjects = os.listdir(subjects_dir)
    patient_dir = st.selectbox("Select a directory", options=subjects, key="patient_dir_aug")
    patient_dir_path = os.path.join(subjects_dir, patient_dir)

    # View, augmentation type, and category selection
    view_categories_aug = ['Axial', 'Sagittal', 'Coronal']
    augment_types = ["None", "Flip", "Rotation", "Scale"]
    selected_view_category = st.selectbox("Select a view category", options=view_categories_aug, index=0, key="view_category_aug")
    augmentation_selection = st.selectbox("Select the type of augmentation", options=augment_types, key="augmentation_selection")

    if st.button("Apply Transformation"):
        categories = ['t1c', 't1n', 't2f', 't2w', 'seg']
        cols = st.columns(len(categories))
        for idx, category in enumerate(categories):
            with cols[idx]:
                st.subheader(f"Category: {category}")
                files_displayed = 0  # Counter to check if any files are displayed
                for file_name in os.listdir(patient_dir_path):
                    if category in file_name and file_name.endswith('.nii.gz'):
                        file_path = os.path.join(patient_dir_path, file_name)
                        sitk_image = load_image(file_path)

                        # Apply transformation
                        sitk_transformed = apply_transformations(sitk_image, augmentation_selection) if augmentation_selection != "None" else sitk_image
                        np_image_original = sitk.GetArrayFromImage(sitk_image)
                        np_image_transformed = sitk.GetArrayFromImage(sitk_transformed)

                        # Get the slice for the selected view
                        slice_index = {
                            'Axial': np_image_original.shape[0] // 2,
                            'Sagittal': np_image_original.shape[2] // 2,
                            'Coronal': np_image_original.shape[1] // 2
                        }[selected_view_category]

                        original_image = np.take(np_image_original, slice_index, axis={'Axial': 0, 'Sagittal': 2, 'Coronal': 1}[selected_view_category])
                        transformed_image = np.take(np_image_transformed, slice_index, axis={'Axial': 0, 'Sagittal': 2, 'Coronal': 1}[selected_view_category])

                        # Create subplot for original and transformed images
                        fig, axes = plt.subplots(1, 2, figsize=(40, 20))  # Increased figure size
                        axes[0].imshow(original_image, cmap='gray')
                        axes[0].set_title('Original Image')
                        axes[0].axis('off')

                        axes[1].imshow(transformed_image, cmap='gray')
                        axes[1].set_title('Transformed Image')
                        axes[1].axis('off')

                        st.pyplot(fig)
                        files_displayed += 1

                if files_displayed == 0:
                    st.write(f"No MRI data available for the category: {category}")


def modelling():
    st.title("Medical Image Segmentation for BRATS 2024")

    # Section Header
    st.header("Modeling Brain Tumors with 3D U-Net and Residual 3D U-Net")

    # Why 3D U-Net?
    st.subheader("Why 3D U-Net for the BRATS 2024 dataset?")
    st.markdown("""
       * **Contextual Preservation**: 
       * **Precise localization**: 
       """)

    # What is U-Net?
    st.subheader("Understanding U-Net Architecture")
    st.write("""
       U-Net architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. The key components include:
       - **Contracting Path**: Captures the context in the image, which helps in understanding the global structure of the brain and the tumor.
       - **Expanding Path**: Allows for precise localization using transposed convolutions to recover spatial resolution.
       - **Skip Connections**: Provide essential high-resolution features to the expanding path, improving the accuracy of segmentation.
       """)

    # Implementation in Code
    st.subheader("Implementing U-Net for BRATS 2024")
    st.code('''class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        ..............
        
        class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        
        
        '''





       , language='python')

    # Metrics Evaluation
    st.subheader("Evaluating Model Performance")
    st.write("""
       Model performance is assessed using the Dice coefficient, which measures the overlap between the predicted segmentation and the ground truth labels. This metric is crucial for understanding the effectiveness of the model in medical segmentation tasks.
       """)
    st.code("""
       def dice_coefficient(preds, targets, num_classes):
           # Compute Dice Coefficient
           ...
           return dice / (num_classes - 1)
       """, language='python')

    # Data Transformation
    st.subheader("Data Preprocessing and Augmentation")
    st.markdown("""
       Effective data preprocessing and augmentation are crucial for training robust models. Here's how data is prepared:
       - **Normalization**: Ensures that MRI scans have similar intensity ranges, enhancing model training stability.
       - **Augmentation**: Includes zooming, cropping, and rotation, which helps the model generalize better by presenting varied examples during training.
       """)
    st.code("""
       transformations = apply_transformations(image)
       """, language='python')

    # Additional Resources
    st.subheader("Additional Resources and References")
    st.markdown("""
       - [Original U-Net Paper](https://arxiv.org/abs/1505.04597)
       - [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
       - [TensorFlow U-Net Tutorial](https://www.tensorflow.org/tutorials/images/segmentation)
       """)

def metric_tab():
    # Load the metrics CSV files from the configuration paths
    resunet_metrics_path = configuration.metrics_resunet  # Path to Residual UNet metrics.csv
    unet_metrics_path = configuration.unet_metrics  # Path to UNet metrics.csv

    # Check if the files exist
    if os.path.exists(resunet_metrics_path):
        resunet_df = pd.read_csv(resunet_metrics_path)
    else:
        st.warning(f"{resunet_metrics_path} does not exist.")
        return

    if os.path.exists(unet_metrics_path):
        unet_df = pd.read_csv(unet_metrics_path)
    else:
        st.warning(f"{unet_metrics_path} does not exist.")
        return

    # Creating a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plotting the Residual UNet Train/Val Loss
    axs[0, 0].plot(resunet_df['epoch'], resunet_df['train_loss'], label='Train Loss', color='blue')
    axs[0, 0].plot(resunet_df['epoch'], resunet_df['val_loss'], label='Val Loss', color='red')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].set_title('Residual UNet Train and Validation Loss')
    axs[0, 0].legend()

    # Plotting the Residual UNet Train/Val Dice Score
    axs[0, 1].plot(resunet_df['epoch'], resunet_df['train_dice'], label='Train Dice Score', color='green')
    axs[0, 1].plot(resunet_df['epoch'], resunet_df['val_dice'], label='Validation Dice Score', color='orange')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Dice Score')
    axs[0, 1].set_title('Residual UNet Train and Validation Dice Score')
    axs[0, 1].legend()

    # Plotting the UNet Train/Val Loss
    axs[1, 0].plot(unet_df['epoch'], unet_df['train_loss'], label='Train Loss', color='blue')
    axs[1, 0].plot(unet_df['epoch'], unet_df['validation_loss'], label='Validation Loss', color='red')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('UNet Train and Validation Loss')
    axs[1, 0].legend()

    # Plotting the UNet Train/Val Dice Score
    axs[1, 1].plot(unet_df['epoch'], unet_df['train_dice'], label='Train Dice Score', color='green')
    axs[1, 1].plot(unet_df['epoch'], unet_df['validation_dice'], label='Validation Dice Score', color='orange')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Dice Score')
    axs[1, 1].set_title('UNet Train and Validation Dice Score')
    axs[1, 1].legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Display the plots
    st.pyplot(fig)

def display_image():
    st.title('Tumor Sub-region Image')

    # Specify the path or URL to your static PNG image
    image_path =configuration.seg_image  # Change this to the path of your local image
    # or
    # image_path = 'https://example.com/your_image.png'  # Direct URL to an image

    # Display the image
    st.image(image_path, caption='Tumor Sub-region Image', use_container_width=True)

def display_image_unet():
    st.title('3D u net Architecture')

    # Specify the path or URL to your static PNG image
    image_path = configuration.unet_image # Change this to the path of your local image
    # or
    # image_path = 'https://example.com/your_image.png'  # Direct URL to an image

    # Display the image
    st.image(image_path, caption='3d-UNET Architecture', use_container_width=True)

def pre_processing():
    st.subheader("Preprocessing  for BRATS 2024")
    st.code('''
    def apply_transformations(image):
    """Apply zooming, cropping, and rotation transformations to the image using SimpleITK."""
    transformations = {}

    # ---- Zooming ----
    zoom_factor = 1.2  # Example: Zoom in by 20%
    size = image.GetSize()
    spacing = image.GetSpacing()
    new_spacing = [s / zoom_factor for s in spacing]  # Decrease spacing to zoom in
    resampler_zoom = sitk.ResampleImageFilter()
    resampler_zoom.SetOutputSpacing(new_spacing)
    resampler_zoom.SetSize([int(size[i] * zoom_factor) for i in range(3)])
    resampler_zoom.SetInterpolator(sitk.sitkLinear)
    resampler_zoom.SetOutputDirection(image.GetDirection())
    resampler_zoom.SetOutputOrigin(image.GetOrigin())
    zoomed_image = resampler_zoom.Execute(image)
    transformations['zoomed'] = zoomed_image

    # ---- Cropping ----
    crop_size = [int(0.8 * s) for s in zoomed_image.GetSize()]  # Crop to 80% of the zoomed image size
    crop_start = [(zoomed_image.GetSize()[i] - crop_size[i]) // 2 for i in range(3)]  # Center cropping
    cropped_image = sitk.RegionOfInterest(zoomed_image, crop_size, crop_start)
    transformations['cropped'] = cropped_image

    # ---- Rotation ----
    transform = sitk.AffineTransform(3)  # 3D affine transformation
    transform.Rotate(0, 1, np.pi / 12)  # Rotate around the z-axis
    resampler_rotate = sitk.ResampleImageFilter()
    resampler_rotate.SetInterpolator(sitk.sitkLinear)
    resampler_rotate.SetTransform(transform)
    resampler_rotate.SetReferenceImage(cropped_image)
    rotated_image = resampler_rotate.Execute(cropped_image)
    transformations['rotated'] = rotated_image

    return transformations


#%%

def resize_image(image, new_size):
    """Resize images to a new size using SimpleITK."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[0] * (image.GetSize()[0] / new_size[0]),
                                image.GetSpacing()[1] * (image.GetSize()[1] / new_size[1]),
                                image.GetSpacing()[2] * (image.GetSize()[2] / new_size[2])])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.AffineTransform(3))
    return resampler.Execute(image)



#%%

def normalize_image(image_np):
    """Normalize image array to have pixel values between 0 and 1."""
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np

    

            '''

            , language='python')


def display_image_unet():
    st.title('3D u net Architecture')

    # Specify the path or URL to your static PNG image
    image_path = configuration.unet_image  # Change this to the path of your local image
    # or
    # image_path = 'https://example.com/your_image.png'  # Direct URL to an image

    # Display the image
    st.image(image_path, caption='3d-UNET Architecture', use_container_width=True)


def stacking():
    st.subheader("Image Resgistration")
    st.code('''
    for mod in self.modalities:
            mod_path = os.path.join(self.image_dir, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)

            # Keep the original image
            original_image = resize_image(image, self.target_size)
            original_image_np = sitk.GetArrayFromImage(original_image)
            normalized_original = normalize_image(original_image_np)
            images_original.append(normalized_original)

            # Apply transformations
            transformations = apply_transformations(image)
            transformed_image = random.choice(list(transformations.values()))
            transformed_image = resize_image(transformed_image, self.target_size)
            transformed_image_np = sitk.GetArrayFromImage(transformed_image)
            normalized_transformed = normalize_image(transformed_image_np)
            images_transformed.append(normalized_transformed)

        # Stack original and transformed images
        image_stack_original = np.stack(images_original, axis=0)
        image_stack_transformed = np.stack(images_transformed, axis=0)



            '''

            , language='python')



st.set_page_config(layout="wide")
Home()
display_image()

visualization()
image_shape()
# pre_processing()
stacking()
aug_visualisation()
modelling()
display_image_unet()
predicting_tab_unet()
predicting_tab_unet_with_transformation()


predicting_tab_residual()
predicting_tab_residual_with_transformation()

metric_tab()





