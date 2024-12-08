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
from model_3dRESN_unet import ResidualUNet3D, Down, Up, OutConv


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
    val_img_dir = '/home/ubuntu/Dataset/val_data/val_images'
    val_mask_dir = '/home/ubuntu/Dataset/val_data/val_masks'

    # Load validation dataset and DataLoader
    val_dataset = BraTSDataset_val(val_img_dir, val_mask_dir)
    val_loader =  DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate1)
    # Initialize model and load state_dict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ResidualUNet3D(n_channels=4, n_classes=5)
    model = model.to(device)
    # Update the path to the ResidualUNet3D checkpoint you saved during training
    state_dict = torch.load('/home/ubuntu/dl_code/models/best_model.pth', map_location=device)
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

