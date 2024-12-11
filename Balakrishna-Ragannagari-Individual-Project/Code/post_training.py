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

def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)

# Helper Function: Apply Transformations
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

# Helper Function: Normalize Image
def normalize_image(image_np):
    """Normalize image array to have pixel values between 0 and 1."""
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np

class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(128,128,128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if any(mod in f for mod in self.modalities) and f.endswith('.nii')]
        )

    def __len__(self):
        return len(self.image_files) // len(self.modalities)

    def __getitem__(self, idx):
        base_filename = self.image_files[idx * len(self.modalities)]
        patient_id = base_filename.split('_')[0]
        images_original = []
        images_transformed = []

        # Process each modality
        for mod in self.modalities:
            mod_path = os.path.join(self.image_dir, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)

            # Keep the original image
            original_image = resize_image(image, self.target_size)
            original_image_np = sitk.GetArrayFromImage(original_image)
            normalized_original = normalize_image(original_image_np)
            images_original.append(normalized_original)

            # Apply transformations
            # transformations = apply_transformations(image)
            # transformed_image = random.choice(list(transformations.values()))
            # transformed_image = resize_image(transformed_image, self.target_size)
            # transformed_image_np = sitk.GetArrayFromImage(transformed_image)
            # normalized_transformed = normalize_image(transformed_image_np)
            # images_transformed.append(normalized_transformed)

        # Stack original and transformed images
        image_stack_original = np.stack(images_original, axis=0)
        # image_stack_transformed = np.stack(images_transformed, axis=-1)

        # Process the segmentation mask
        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        # Keep the original segmentation
        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)

        # Set label 4 to 0 directly
        # original_seg_array[original_seg_array == 4] = 0
        # unique_original = np.unique(original_seg_array)
        # print(f"Unique values in the original segmentation: {unique_original}")

        # Apply the same transformation to the segmentation
        # segmentation_transformations = apply_transformations(segmentation)
        # transformed_segmentation = random.choice(list(segmentation_transformations.values()))
        # transformed_segmentation = resize_image(transformed_segmentation, self.target_size)
        # transformed_seg_array = sitk.GetArrayFromImage(transformed_segmentation)

        # Set label 4 to 0 in the transformed segmentation
        # transformed_seg_array[transformed_seg_array == 4] = 0
        # unique_transformed = np.unique(transformed_seg_array)
        # print(f"Unique values in the transformed segmentation: {unique_transformed}")

        # Return both original and transformed versions
        return (
            torch.tensor(image_stack_original, dtype=torch.float32),
            torch.tensor(original_seg_array, dtype=torch.long),
        )

def custom_collate(batch):
    images_original, masks_original = zip(*batch)
    return (
        torch.stack(images_original),
        torch.stack(masks_original),
    )

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, try using reflect padding as an alternative
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

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





# Visualization Functions
def visualize_batch(images, segmentations):
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 6 * batch_size))
    for i in range(batch_size):
        axes[i, 0].imshow(images[i, :, :, images.shape[2] // 2, 0], cmap='gray')
        axes[i, 0].set_title('Transformed and Normalized Image (t1c)')
        axes[i, 1].imshow(segmentations[i, :, :, segmentations.shape[2] // 2], cmap='gray')
        axes[i, 1].set_title('Transformed Segmentation Mask')
    plt.show()


#%%-----------------------
import torch

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

def evaluate_model(model, data_loader, criterion, num_classes, device='cuda'):
    model.eval()
    total_loss = 0
    total_dice_scores = [0] * (num_classes - 1)  # Exclude background (class 0)

    with torch.no_grad():
        for images, segmentations in data_loader:
            images = images.to(device)
            segmentations = segmentations.to(device)

            # Forward pass
            images = images # Ensure channel-first format
            outputs = model(images)
            loss = criterion(outputs, segmentations)
            total_loss += loss.item()

            # Compute Dice scores
            dice_scores = dice_coefficient(outputs, segmentations, num_classes)
            total_dice_scores = [total_dice_scores[i] + dice_scores[i] for i in range(len(dice_scores))]

    # Normalize Dice scores across all batches
    total_dice_scores = [score / len(data_loader) for score in total_dice_scores]

    print(f"Validation Loss: {total_loss / len(data_loader):.4f}")
    print(f"Validation Dice Scores (Class 1-3): {total_dice_scores}")
    return total_loss / len(data_loader), total_dice_scores


def visualize_predictions_temp(images, targets, predictions, slice_idx=None):
    """
    Visualize the input image, ground truth, and predictions for a specific depth slice.

    Args:
        images (numpy.ndarray): Input images, shape [Batch, Channels, Depth, Height, Width].
        targets (numpy.ndarray): Ground truth masks, shape [Batch, Depth, Height, Width].
        predictions (numpy.ndarray): Predicted masks, shape [Batch, Depth, Height, Width].
        slice_idx (int, optional): Slice index along the depth dimension. Defaults to the middle slice.
    """
    # Get the slice index (middle slice if not specified)
    if slice_idx is None:
        slice_idx = images.shape[2] // 2  # Assuming the depth is at index 2 for images

    for i in range(images.shape[0]):  # Iterate over the batch
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Extract the slice for visualization
        input_image = images[i, 0, slice_idx, :, :]  # Use the first channel (e.g., T1c)
        target_mask = targets[i, slice_idx, :, :]  # Ground truth segmentation
        predicted_mask = predictions[i, slice_idx, :, :]  # Predicted segmentation

        # Plot the input image
        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Input Image (Slice)')

        # Plot the ground truth segmentation mask
        axes[1].imshow(target_mask, cmap='gray')
        axes[1].set_title('Ground Truth (Slice)')

        # Plot the predicted segmentation mask
        axes[2].imshow(predicted_mask, cmap='gray')
        axes[2].set_title('Prediction (Slice)')

        plt.tight_layout()
        plt.show()

def create_custom_colormap():
    # Define colors for each label
    colors = [(0, 0, 0),         # Background - black
              (1, 0, 0),         # Necrotic - red
              (0, 1, 0),         # Edema - green
              (0, 0, 1),         # Enhancing - blue
              (1, 1, 0)]         # Fifth Label - yellow (example color, adjust as needed)
    cmap_name = 'tumor_segmentation'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=len(colors))

def visualize_predictions(images, targets, predictions, slice_idx=None):
    """
    Visualize the input image, ground truth, and predictions for a specific depth slice
    with each segmentation label shown in a different color and labels annotated.

    Args:
        images (numpy.ndarray): Input images, shape [Batch, Channels, Depth, Height, Width].
        targets (numpy.ndarray): Ground truth masks, shape [Batch, Depth, Height, Width].
        predictions (numpy.ndarray): Predicted masks, shape [Batch, Depth, Height, Width].
        slice_idx (int, optional): Slice index along the depth dimension. Defaults to the middle slice.
    """
    cmap = create_custom_colormap()  # Custom colormap for the labels
    label_descriptions = {
        0: 'Background',
        1: 'Necrotic/Core',
        2: 'Edema',
        3: 'Enhancing Tumor',
        4: 'Resection Cavity'  # Update this as per your dataset specifics
    }

    if slice_idx is None:
        slice_idx = images.shape[2] // 2  # Assuming the depth is at index 2 for images

    for i in range(images.shape[0]):  # Iterate over the batch
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = [cmap(i) for i in range(len(label_descriptions))]

        input_image = images[i, 0, slice_idx, :, :]
        target_mask = targets[i, slice_idx, :, :]
        predicted_mask = predictions[i, slice_idx, :, :]

        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Input Image (Slice)')
        im1 = axes[1].imshow(target_mask, cmap=cmap)
        axes[1].set_title('Ground Truth (Slice)')
        im2 = axes[2].imshow(predicted_mask, cmap=cmap)
        axes[2].set_title('Prediction (Slice)')

        # Create a legend for the colors indicating each tissue type
        patches = [plt.Line2D([0], [0], color=colors[idx], lw=4, label=label_descriptions[idx]) for idx in label_descriptions]
        plt.figlegend(patches, [label_descriptions[idx] for idx in label_descriptions], loc='lower center', ncol=3, labelspacing=0.)

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Paths for validation data
    val_img_dir = '/home/ubuntu/Dataset/val_data/val_images'
    val_mask_dir = '/home/ubuntu/Dataset/val_data/val_masks'

    # Load validation dataset and DataLoader
    val_dataset = BraTSDataset(val_img_dir, val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # Initialize model and load state_dict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet3D(n_channels=4, n_classes=5)
    model = model.to(device)
    state_dict = torch.load('/home/ubuntu/DL_Project5/testting_project_group5/models/model_epoch_9.pth')
    model.load_state_dict(state_dict)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()  # Define the loss function
    num_classes = 5  # Total number of classes (including background)

    # Evaluate the model
    val_loss, val_dice_scores = evaluate_model(model, val_loader, criterion, num_classes, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Dice Scores: {val_dice_scores}")

    model.eval()
    c=4
    with torch.no_grad():
        for images, segmentations in val_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            visualize_predictions(images.cpu().numpy(), segmentations.numpy(), predictions.cpu().numpy())
            if(c==10):
                break
            c=c+1
