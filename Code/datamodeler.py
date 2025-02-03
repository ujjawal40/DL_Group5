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


# Normalize image array to have pixel values between 0 and 1
def normalize_image(image_np):
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np


def apply_transformations(image):
    """Apply zooming, cropping, and rotation transformations to the image using SimpleITK and return the last transformed image."""

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

    # ---- Cropping ----
    crop_size = [int(0.8 * s) for s in zoomed_image.GetSize()]  # Crop to 80% of the zoomed image size
    crop_start = [(zoomed_image.GetSize()[i] - crop_size[i]) // 2 for i in range(3)]  # Center cropping
    cropped_image = sitk.RegionOfInterest(zoomed_image, crop_size, crop_start)

    # ---- Rotation ----
    transform = sitk.AffineTransform(3)  # 3D affine transformation
    transform.Rotate(0, 1, np.pi / 12)  # Rotate around the z-axis
    resampler_rotate = sitk.ResampleImageFilter()
    resampler_rotate.SetInterpolator(sitk.sitkLinear)
    resampler_rotate.SetTransform(transform)
    resampler_rotate.SetReferenceImage(cropped_image)
    rotated_image = resampler_rotate.Execute(cropped_image)

    return rotated_image
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

            transformed_image = apply_transformations(image)

            # Normalize the transformed image
            normalized_transformed = normalize_image(transformed_image)
            images_transformed.append(normalized_transformed)


            # In either case, we only need to store the normalized original image


        image_stack_original = np.stack(images_original, axis=0)
        image_stack_transformed = np.stack(images_transformed, axis=0)

        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)

        transformed_segmentation = apply_transformations(segmentation)
        transformed_segmentation = resize_image(transformed_segmentation, self.target_size)
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




        image_stack_original = np.stack(images_original, axis=0)


        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)


        return (
            torch.tensor(image_stack_original, dtype=torch.float32),
            torch.tensor(original_seg_array, dtype=torch.long),
        )

def custom_collate1(batch):
    images_original, masks_original = zip(*batch)
    return (
        torch.stack(images_original),
        torch.stack(masks_original)
    )


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

class TransUNet(nn.Module):
    def __init__(self, in_channels, n_classes, img_size, patch_size, embed_dim, num_heads, num_layers):
        super(TransUNet, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch Embedding and Transformer Encoder
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)

        # Decoder
        self.up1 = Up(embed_dim, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x_patches = self.patch_embedding(x)
        x_transformer = self.transformer_encoder(x_patches)

        # Reshape back to spatial dimensions
        D_p, H_p, W_p = D // self.patch_size, H // self.patch_size, W // self.patch_size
        x_transformer = x_transformer.transpose(1, 2).view(B, -1, D_p, H_p, W_p)

        # Decode
        x = self.up1(x_transformer, None)  # Skip connections can be added here
        x = self.up2(x, None)
        x = self.up3(x, None)
        logits = self.outc(x)
        return logits

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(x1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#%%

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_directory, device):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    best_val_dice = -float('inf')
    model.to(device)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_dice = 0.0
        batch_start_time = time.time()  # Initialize batch start time

        # Training phase
        for batch_index, (images, _, images_transformed, masks_transformed) in enumerate(train_loader):
            images, masks = images_transformed.to(device), masks_transformed.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_dice += dice_coefficient(torch.argmax(outputs, dim=1), masks, 5)

            # Print debug information for each batch
            if batch_index % 10 == 0:  # Print every 10 batches, adjust this number based on your batch size and preferences
                batch_end_time = time.time()
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_index + 1}/{len(train_loader)}, '
                      f'Batch Loss: {loss.item():.4f}, Batch Dice: {dice_coefficient(torch.argmax(outputs, dim=1), masks, 5):.4f}, '
                      f'Batch Time: {batch_end_time - batch_start_time:.2f}s')
                batch_start_time = time.time()  # Reset batch start time

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = train_dice / len(train_loader)

        # Validation phase
        val_loss = 0.0
        val_dice = 0.0
        model.eval()  # Set model to evaluate mode
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                val_dice += dice_coefficient(torch.argmax(outputs, dim=1), masks, 5)

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Dice: {epoch_dice:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_path = os.path.join(save_directory, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved at {best_model_path} with Val Dice: {best_val_dice:.4f}')

        # Save model checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth'))
        print(f'Epoch {epoch + 1} model saved. Time per epoch: {(time.time() - start_time):.2f}s')


def dice_coefficient(preds, targets, num_classes):
    """Calculate the Dice coefficient for batch of predictions and targets"""
    dice = 0
    for class_index in range(1, num_classes):  # Exclude background
        pred = (preds == class_index).float()
        target = (targets == class_index).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + 1e-6
        dice += (2. * intersection + 1e-6) / union
    return dice / (num_classes - 1)


#%%

img_dir = '/home/ubuntu/Dataset/combined_data/train_images'
mask_dir = '/home/ubuntu/Dataset/combined_data/masks'

val_img_dir = '/home/ubuntu/Dataset/val_data/val_images'
val_mask_dir = '/home/ubuntu/Dataset/val_data/val_masks'


dataset = BraTSDataset(img_dir, mask_dir)

val_dataset=BraTSDataset_val(val_img_dir, val_mask_dir)

data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=custom_collate1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransUNet(
    in_channels=4, n_classes=5, img_size=(132, 132, 116), patch_size=(16, 16, 16),
    embed_dim=768, num_heads=8, num_layers=12
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

save_directory='/home/ubuntu/DL_Project5/testting_project_group5/trans-unet-models'
num_epochs=10
train_model(model, data_loader, val_loader, criterion, optimizer, num_epochs, save_directory, device)





