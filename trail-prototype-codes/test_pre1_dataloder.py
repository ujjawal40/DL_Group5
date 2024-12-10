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

# Helper Function: Load Image
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

# Helper Function: Resize Image
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

# Dataset Class
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
            transformations = apply_transformations(image)
            transformed_image = random.choice(list(transformations.values()))  # need to apply transformation not by random
            transformed_image = resize_image(transformed_image, self.target_size)
            transformed_image_np = sitk.GetArrayFromImage(transformed_image)
            normalized_transformed = normalize_image(transformed_image_np)
            images_transformed.append(normalized_transformed)

        # Stack original and transformed images
        image_stack_original = np.stack(images_original, axis=-1)
        image_stack_transformed = np.stack(images_transformed, axis=-1)

        # Process the segmentation mask
        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)

        # Keep the original segmentation
        original_segmentation = resize_image(segmentation, self.target_size)
        original_seg_array = sitk.GetArrayFromImage(original_segmentation)
        processed_segmentation = np.zeros(original_seg_array.shape, dtype=np.int64)
        processed_segmentation[original_seg_array == 3] = 1  # Enhancing Tumor
        processed_segmentation[np.isin(original_seg_array, [1, 3])] = 2  # Tumor Core
        processed_segmentation[np.isin(original_seg_array, [1, 2, 3])] = 3

        # Apply the same transformation to the segmentation
        segmentation_transformations = apply_transformations(segmentation)
        transformation_index = random.randint(0, len(segmentation_transformations) - 1)
        transformed_segmentation = list(segmentation_transformations.values())[transformation_index]
        transformed_segmentation = resize_image(transformed_segmentation, self.target_size)
        transformed_seg_array = sitk.GetArrayFromImage(transformed_segmentation)

        processed_transformed_segmentation = np.zeros(transformed_seg_array.shape, dtype=np.int64)
        processed_transformed_segmentation[transformed_seg_array == 3] = 1  # Enhancing Tumor
        processed_transformed_segmentation[np.isin(transformed_seg_array, [1, 3])] = 2  # Tumor Core
        processed_transformed_segmentation[np.isin(transformed_seg_array, [1, 2, 3])] = 3  # Whole Tumor

        # Return both original and transformed versions
        return (
            torch.tensor(image_stack_original, dtype=torch.float32),
            torch.tensor(processed_segmentation, dtype=torch.long),
            torch.tensor(image_stack_transformed, dtype=torch.float32),
            torch.tensor(processed_transformed_segmentation, dtype=torch.long)
        )


# Collate Function for DataLoader
def custom_collate(batch):
    images_original, masks_original, images_transformed, masks_transformed = zip(*batch)
    return (
        torch.stack(images_original),
        torch.stack(masks_original),
        torch.stack(images_transformed),
        torch.stack(masks_transformed)
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)  # Upsampling from bottleneck
        self.dec4 = self.double_conv(512 + 512, 512)

        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)  # Upsampling from dec4
        self.dec3 = self.double_conv(256 + 256, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)  # Upsampling from dec3
        self.dec2 = self.double_conv(128 + 128, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # Upsampling from dec2
        self.dec1 = self.double_conv(64 + 64, 64)

        # Final Convolution
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [Batch, 64, Depth, Height, Width]
        enc2 = self.enc2(self.pool(enc1))  # [Batch, 128, Depth/2, Height/2, Width/2]
        enc3 = self.enc3(self.pool(enc2))  # [Batch, 256, Depth/4, Height/4, Width/4]
        enc4 = self.enc4(self.pool(enc3))  # [Batch, 512, Depth/8, Height/8, Width/8]

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # [Batch, 1024, Depth/16, Height/16, Width/16]

        # Decoder
        up4 = self.up4(bottleneck)  # [Batch, 512, Depth/8, Height/8, Width/8]
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))  # [Batch, 512, Depth/8, Height/8, Width/8]

        up3 = self.up3(dec4)  # [Batch, 256, Depth/4, Height/4, Width/4]
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))  # [Batch, 256, Depth/4, Height/4, Width/4]

        up2 = self.up2(dec3)  # [Batch, 128, Depth/2, Height/2, Width/2]
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))  # [Batch, 128, Depth/2, Height/2, Width/2]

        up1 = self.up1(dec2)  # [Batch, 64, Depth, Height, Width]
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))  # [Batch, 64, Depth, Height, Width]

        return self.final_conv(dec1)  # [Batch, out_channels, Depth, Height, Width]


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


def train_model(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    best_model = None
    best_metric = float('-inf')  # Assuming higher Dice scores are better

    for epoch in range(num_epochs):
        print(f"\n=== Starting Epoch [{epoch + 1}/{num_epochs}] ===")  # Debug: Starting Epoch
        epoch_loss = 0
        epoch_dice_scores = [0] * (4 - 1)  # 4 classes, excluding background

        # Process batches
        for batch_idx, (original_images, original_masks, transformed_images, transformed_masks) in enumerate(
                data_loader):
            print(f"Processing Batch [{batch_idx + 1}/{len(data_loader)}]")  # Debug: Processing Batch

            # Combine original and transformed data
            inputs = torch.cat([original_images, transformed_images], dim=0).to(device)
            targets = torch.cat([original_masks, transformed_masks], dim=0).to(device)

            # Forward Pass
            inputs = inputs.permute(0, 4, 1, 2, 3)  # Rearrange dimensions for 3D U-Net
            outputs = model(inputs)  # Shape: [Batch Size * 2, Classes, Depth, Height, Width]
            loss = criterion(outputs, targets)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Epoch Loss
            epoch_loss += loss.item()

            # Compute Dice coefficient for this batch
            batch_dice_scores = dice_coefficient(outputs, targets, num_classes=4)
            epoch_dice_scores = [epoch_dice_scores[i] + batch_dice_scores[i] for i in range(len(batch_dice_scores))]

        # Average Dice scores for the epoch
        epoch_dice_scores = [score / len(data_loader) for score in epoch_dice_scores]

        # Debug: End of Epoch Summary
        print(f"\n=== End of Epoch [{epoch + 1}/{num_epochs}] ===")
        print(f"Epoch Loss: {epoch_loss / len(data_loader):.4f}")
        print(f"Epoch Dice Scores (Class 1-3): {epoch_dice_scores}")

        # Save the model after every epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"Saved model for epoch {epoch + 1}.")

        # Save the best model based on the average Dice score across all non-background classes
        avg_dice_score = sum(epoch_dice_scores) / len(epoch_dice_scores)
        if avg_dice_score > best_metric:
            best_metric = avg_dice_score
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")
            print(f"Saved the best model with Dice Score: {best_metric:.4f}.")

    print(f"Training complete. Best model Dice Score: {best_metric:.4f}")
# Evaluation Loop
def evaluate_model_with_dice(model, data_loader, criterion, num_classes):
    model.eval()
    total_loss = 0
    total_dice_scores = [0] * (num_classes - 1)  # Exclude background

    with torch.no_grad():
        for original_images, original_masks, transformed_images, transformed_masks in data_loader:
            inputs = torch.cat([original_images, transformed_images], dim=0).to(device)
            targets = torch.cat([original_masks, transformed_masks], dim=0).to(device)

            # Forward Pass
            inputs = inputs.permute(0, 4, 1, 2, 3)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Compute Dice scores
            batch_dice_scores = dice_coefficient(outputs, targets, num_classes)
            total_dice_scores = [total_dice_scores[i] + batch_dice_scores[i] for i in range(len(batch_dice_scores))]
    total_dice_scores = [score / len(data_loader) for score in total_dice_scores]

    print(f"Validation Loss: {total_loss / len(data_loader):.4f}")
    print(f"Validation Dice Scores (Class 1-3): {total_dice_scores}")
    return total_dice_scores

#%%--------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D(in_channels=4, out_channels=4).to(device)  # 4 modalities, 5 segmentation classes

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 5
batch_size = 1
# Main Script
img_dir = '/home/ubuntu/Dataset/combined_data/train_images'
mask_dir = '/home/ubuntu/Dataset/combined_data/masks'

# Create Dataset and DataLoader
dataset = BraTSDataset(img_dir, mask_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate)
#
# train_model(model, data_loader, criterion, optimizer, num_epochs)

# Evaluate the Model
# evaluate_model_with_dice(model, data_loader, criterion,4)

# Save the Model


# Visualize Batch
# for images, segmentations in data_loader:
#     visualize_batch(images, segmentations)
#     break
