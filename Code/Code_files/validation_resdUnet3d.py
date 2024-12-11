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


def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)

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


def normalize_image(image_np):
    """Normalize image array to have pixel values between 0 and 1."""
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np


class BraTSDataset_val(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(132, 132, 116)):
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

        # Process each modality
        for mod in self.modalities:
            mod_path = os.path.join(self.image_dir, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)

            # Keep the original image
            original_image = resize_image(image, self.target_size)
            original_image_np = sitk.GetArrayFromImage(original_image)
            normalized_original = normalize_image(original_image_np)
            images_original.append(normalized_original)

        image_stack_original = np.stack(images_original, axis=0)

        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)

        return (
            torch.tensor(image_stack_original, dtype=torch.float32),
            torch.tensor(original_seg_array, dtype=torch.long),
        )

# Collate Function for DataLoader

def custom_collate1(batch):
    images_original, masks_original = zip(*batch)
    return (
        torch.stack(images_original),
        torch.stack(masks_original)
    )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Skip connection: if input channels and output channels are different, use a 1x1 convolution to match dimensions
        self.skip_connection = nn.Conv3d(in_channels, out_channels,
                                         kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        if self.skip_connection:
            identity = self.skip_connection(x)

        out += identity  # Add skip connection
        out = self.relu(out)
        return out


class ResidualUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResidualUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ResidualBlock(n_channels, 64)
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

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then ResidualBlock"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels)  # Remove in_channels // 2
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels // 2, out_channels)  # Corrected here as well

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




#%%

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
    with each segmentation label shown in a different color.

    Args:
        images (numpy.ndarray): Input images, shape [Batch, Channels, Depth, Height, Width].
        targets (numpy.ndarray): Ground truth masks, shape [Batch, Depth, Height, Width].
        predictions (numpy.ndarray): Predicted masks, shape [Batch, Depth, Height, Width].
        slice_idx (int, optional): Slice index along the depth dimension. Defaults to the middle slice.
    """
    cmap = create_custom_colormap()  # Custom colormap for the labels

    if slice_idx is None:
        slice_idx = images.shape[2] // 2  # Assuming the depth is at index 2 for images

    for i in range(images.shape[0]):  # Iterate over the batch
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        input_image = images[i, 0, slice_idx, :, :]
        target_mask = targets[i, slice_idx, :, :]
        predicted_mask = predictions[i, slice_idx, :, :]

        axes[0].imshow(input_image, cmap='gray')
        axes[0].set_title('Input Image (Slice)')

        axes[1].imshow(target_mask, cmap=cmap)
        axes[1].set_title('Ground Truth (Slice)')

        axes[2].imshow(predicted_mask, cmap=cmap)
        axes[2].set_title('Prediction (Slice)')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Paths for validation data
    # Get the directory in which the current script is located
    current_script_path = os.path.dirname(os.path.abspath(__file__))

    # Navigate up one level to the parent folder, which is DL-PROJECT
    parent_dir = os.path.dirname(current_script_path)

    val_img_dir = os.path.join(parent_dir, 'validation_data', 'validation_images')
    val_mask_dir = os.path.join(parent_dir, 'validation_data', 'validation_masks')

    # Load validation dataset and DataLoader
    val_dataset = BraTSDataset_val(val_img_dir, val_mask_dir)
    val_loader =  DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate1)
    # Initialize model and load state_dict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResidualUNet3D(n_channels=4, n_classes=5)
    model = model.to(device)
    # Update the path to the ResidualUNet3D checkpoint you saved during training
    model_dir = os.path.join(parent_dir, 'models', 'models_by_train')
    model_path = f'{model_dir}/best_model_residualUnet3d.pth'
    state_dict = torch.load(model_path, map_location=device)
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
