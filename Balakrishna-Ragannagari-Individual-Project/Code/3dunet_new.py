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


#%%
def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)


#%%
def apply_transformations(image):
    """Apply zooming, cropping, rotation, and random flipping transformations to the image."""

    # Zooming
    zoom_factor = 1.2
    size = image.GetSize()
    spacing = image.GetSpacing()
    new_spacing = [s / zoom_factor for s in spacing]
    resampler_zoom = sitk.ResampleImageFilter()
    resampler_zoom.SetOutputSpacing(new_spacing)
    resampler_zoom.SetSize([int(size[i] * zoom_factor) for i in range(3)])
    resampler_zoom.SetInterpolator(sitk.sitkLinear)
    resampler_zoom.SetOutputDirection(image.GetDirection())
    resampler_zoom.SetOutputOrigin(image.GetOrigin())
    zoomed_image = resampler_zoom.Execute(image)

    # Cropping
    crop_size = [int(0.8 * s) for s in zoomed_image.GetSize()]
    crop_start = [(zoomed_image.GetSize()[i] - crop_size[i]) // 2 for i in range(3)]
    cropped_image = sitk.RegionOfInterest(zoomed_image, crop_size, crop_start)

    # Rotation
    transform = sitk.AffineTransform(3)
    transform.Rotate(0, 1, np.pi / 12)
    resampler_rotate = sitk.ResampleImageFilter()
    resampler_rotate.SetInterpolator(sitk.sitkLinear)
    resampler_rotate.SetTransform(transform)
    resampler_rotate.SetReferenceImage(cropped_image)
    rotated_image = resampler_rotate.Execute(cropped_image)

    # Random flipping
    flip_axes = [False, False, False]  # Flip axes; can be randomly set for x, y, or z axes
    if np.random.rand() > 0.5:  # Randomly decide to flip horizontally
        flip_axes[0] = True
    if np.random.rand() > 0.5:  # Randomly decide to flip vertically
        flip_axes[1] = True

    flip_transform = sitk.FlipImageFilter()
    flip_transform.SetFlipAxes(flip_axes)
    flipped_image = flip_transform.Execute(rotated_image)

    return flipped_image


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



#%%

class BraTSDataset(Dataset):
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

            # Apply transformations and select one transformation (for example, 'zoomed')
            transformed_image = apply_transformations(image)  # You can also choose 'cropped' or 'rotated'
            transformed_image_np = sitk.GetArrayFromImage(transformed_image)
            normalized_transformed = normalize_image(transformed_image_np)
            images_transformed.append(normalized_transformed)

        image_stack_original = np.stack(images_original, axis=0)
        image_stack_transformed = np.stack(images_transformed, axis=0)

        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)

        # Apply the same transformation to the segmentation (select one transformation)
        transformed_segmentation = apply_transformations(segmentation)
        transformed_segmentation = transformed_segmentation  # Choose the same transformation as above
        transformed_seg_array = sitk.GetArrayFromImage(transformed_segmentation)

        return (
            torch.tensor(image_stack_original, dtype=torch.float32),
            torch.tensor(original_seg_array, dtype=torch.long),
            torch.tensor(image_stack_transformed, dtype=torch.float32),
            torch.tensor(transformed_seg_array, dtype=torch.long)
        )


def custom_collate(batch):
    images_original, masks_original, images_transformed, masks_transformed = zip(*batch)
    return (
        torch.stack(images_original),
        torch.stack(masks_original),
        torch.stack(images_transformed),
        torch.stack(masks_transformed)
    )




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



#%%%%


def dice_coefficient(preds, targets, num_classes):
    dice = 0.0
    # Loop over each class except background
    for class_index in range(1, num_classes):  # start from 1 to exclude background
        pred = (preds == class_index).float()
        target = (targets == class_index).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + 1e-6
        dice += (2. * intersection + 1e-6) / union

    return dice / (num_classes - 1)


#%%------------------------------------------------------------------
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_directory, patience=5,
#                 num_classes=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     best_val_dice = 0.0
#     early_stopping_counter = 0
#
#     # Initialize GradScaler for mixed precision
#     scaler = GradScaler()
#
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss, train_dice = 0, 0
#         total_batches = len(train_loader)
#
#         for batch_idx, data in enumerate(train_loader):
#             # Unpack all four items; use only the first and second for training
#             images, masks, _, _ = data
#             images, masks = images.to(device), masks.to(device)
#
#             optimizer.zero_grad()
#
#             # Use autocast for mixed precision
#             with autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#
#             # Scale the loss for backpropagation
#             scaler.scale(loss).backward()
#
#             # Optimizer step (with mixed precision scaling)
#             scaler.step(optimizer)
#
#             # Update the scaler
#             scaler.update()
#
#             train_loss += loss.item()
#
#             # Predictions and Dice score calculation
#             preds = torch.argmax(outputs, dim=1)  # Get the index of the max log-probability
#             dice_score = dice_coefficient(preds, masks, num_classes).item()
#             train_dice += dice_score
#
#             # Debugging output for each batch
#             print(
#                 f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}, Dice Score: {dice_score:.4f}')
#
#         train_loss /= len(train_loader)
#         train_dice /= len(train_loader)
#
#         # Validation phase
#         model.eval()
#         val_loss, val_dice = 0, 0
#         with torch.no_grad():
#             for batch_idx, data in enumerate(val_loader):
#                 images, masks, _, _ = data
#                 images, masks = images.to(device), masks.to(device)
#
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#                 val_loss += loss.item()
#
#                 preds = torch.argmax(outputs, dim=1)
#                 val_dice += dice_coefficient(preds, masks, num_classes).item()
#
#         val_loss /= len(val_loader)
#         val_dice /= len(val_loader)
#
#         print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')
#
#         # Save model after each epoch
#         if val_dice > best_val_dice:
#             best_val_dice = val_dice
#             torch.save(model.state_dict(), os.path.join(save_directory, 'best_model.pth'))
#             print("Best model saved with Dice:", best_val_dice)
#             early_stopping_counter = 0
#         else:
#             early_stopping_counter += 1
#             print(f"No improvement in validation, early stopping counter: {early_stopping_counter}")
#             if early_stopping_counter >= patience:
#                 print("Early stopping triggered.")
#                 break
#
#         # Save model for each epoch
#         torch.save(model.state_dict(), os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth'))
#         print(f"Model saved for epoch {epoch + 1}.")
#

#%%
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_directory, patience=5, num_classes=5):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     best_val_dice = 0.0
#     early_stopping_counter = 0
#
#     # Initialize GradScaler for mixed precision
#     scaler = GradScaler()
#
#     train_losses, val_losses, train_dice_scores, val_dice_scores = [], [], [], []
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss, total_train_dice = 0, 0
#
#         for batch_idx, (images, masks, _, _) in enumerate(train_loader):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad()
#
#             # Autocast to mixed precision
#             with autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#
#             # Backward pass with automatic mixed precision scaling
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#
#             preds = torch.argmax(outputs, dim=1)
#             dice_score = dice_coefficient(preds, masks, num_classes)
#
#             total_train_loss += loss.item()
#             total_train_dice += dice_score.item()  # Convert to Python float
#             print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Dice: {dice_score.item():.4f}')
#
#
#
#         avg_train_loss = total_train_loss / len(train_loader.dataset)
#         avg_train_dice = total_train_dice / len(train_loader.dataset)
#         train_losses.append(avg_train_loss)
#         train_dice_scores.append(avg_train_dice)
#
#         model.eval()
#         total_val_loss, total_val_dice = 0, 0
#         with torch.no_grad():
#             c=0
#             for images, masks in val_loader:
#                 images, masks = images.to(device), masks.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, masks)
#                 preds = torch.argmax(outputs, dim=1)
#                 dice_score = dice_coefficient(preds, masks, num_classes)
#
#                 total_val_loss += loss.item()
#                 total_val_dice += dice_score.item()
#                 print(c)
#                 c=c+1
#
#         avg_val_loss = total_val_loss / len(val_loader.dataset)
#         avg_val_dice = total_val_dice / len(val_loader.dataset)
#         val_losses.append(avg_val_loss)
#         val_dice_scores.append(avg_val_dice)
#
#         print(f'Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}')
#         epoch_save_path = os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth')
#         torch.save(model.state_dict(), epoch_save_path)
#         print(f"Saved epoch {epoch + 1} model checkpoint at {epoch_save_path}.")
#
#         if avg_val_dice > best_val_dice:
#             best_val_dice = avg_val_dice
#             torch.save(model.state_dict(), os.path.join(save_directory, 'best_model.pth'))
#             early_stopping_counter = 0
#         else:
#             early_stopping_counter += 1
#             if early_stopping_counter >= patience:
#                 print("Early stopping triggered.")
#                 break
#
#     # Store metrics to DataFrame and save
#     history_df = pd.DataFrame({
#         'Train Loss': train_losses,
#         'Validation Loss': val_losses,
#         'Train Dice': train_dice_scores,
#         'Validation Dice': val_dice_scores
#     })
#     history_df.to_csv(os.path.join(save_directory, 'training_history.csv'), index=False)
#     return history_df

#%%

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_directory, num_classes, patience=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Lists to hold all metrics for training and validation
    train_losses, val_losses, train_dice_scores, val_dice_scores = [], [], [], []
    best_val_dice = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_train_dice = 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Autocast to mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            dice_score = dice_coefficient(preds, masks, num_classes)

            total_train_loss += loss.item()
            total_train_dice += dice_score.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_dice = total_train_dice / len(train_loader)
        train_losses.append(avg_train_loss)
        train_dice_scores.append(avg_train_dice)

        # Validation phase
        model.eval()
        total_val_loss, total_val_dice = 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                dice_score = dice_coefficient(preds, masks, num_classes)

                total_val_loss += loss.item()
                total_val_dice += dice_score.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_dice = total_val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dice_scores.append(avg_val_dice)

        # Print epoch metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Training Loss: {avg_train_loss:.4f}, Training Dice: {avg_train_dice:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

        # Check for early stopping
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            early_stopping_counter = 0
            # Save the best model
            best_model_path = os.path.join(save_directory, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved with Dice:", best_val_dice)
        else:
            early_stopping_counter += 1
            print(f"No improvement in validation, early stopping counter: {early_stopping_counter}")
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save model for each epoch regardless of improvement
        epoch_save_path = os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_save_path)

    # Save the metrics to a DataFrame and then to CSV
    history_df = pd.DataFrame({
        'Train Loss': train_losses,
        'Train Dice': train_dice_scores,
        'Validation Loss': val_losses,
        'Validation Dice': val_dice_scores
    })
    history_df.to_csv(os.path.join(save_directory, 'training_metrics.csv'), index=False)

    return history_df

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dir = '/home/ubuntu/Dataset/combined_data/train_images'
mask_dir = '/home/ubuntu/Dataset/combined_data/masks'

val_img_dir = '/home/ubuntu/Dataset/val_data/val_images'
val_mask_dir = '/home/ubuntu/Dataset/val_data/val_masks'


dataset = BraTSDataset(img_dir, mask_dir)

# val_dataset=BraTSDataset_val(val_img_dir, val_mask_dir)

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate)

val_dataset = BraTSDataset_val(val_img_dir, val_mask_dir)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate1)
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,collate_fn=custom_collate1)


sav_dir='/home/ubuntu/DL_Project5/testting_project_group5/new_3dunet_models'
# model = UNet3D(n_channels=4, n_classes=5)  # Adjust the number of channels and classes based on your dataset specifics
model = UNet3D(n_channels=4, n_classes=5)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # You might adjust this depending on your specific task
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

history = train_model(model, data_loader, criterion, optimizer, 10, sav_dir,5)


history_df=pd.DataFrame(history)
history_df.to_csv('/home/ubuntu/DL_Project5/testting_project_group5/new_3dunet_models/metrics.csv', index=False)
