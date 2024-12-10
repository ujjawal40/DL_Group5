import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F
import os
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_coeff(pred, target, num_classes=4, smooth=1e-6):
    """
    Compute the Dice coefficient for each class separately in a multi-class segmentation task.
    Args:
        pred: predicted tensor (logits or probabilities)
        target: ground truth tensor (long integers representing class labels)
        num_classes: number of classes (default is 4)
        smooth: small constant to avoid division by zero
    Returns:
        A list of Dice coefficients for each class
    """
    # Ensure pred is of float type for softmax
    pred = pred.float()

    # Apply softmax to convert logits to probabilities
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities

    dice_scores = []

    # Calculate Dice coefficient for each class separately
    for i in range(num_classes):
        # Create binary masks for the current class
        pred_class = (pred.argmax(dim=1) == i).float()  # Predicted class mask (after applying argmax)
        target_class = (target == i).float()  # Ground truth class mask

        # Flatten both tensors to compute the intersection and union
        pred_class = pred_class.view(-1)
        target_class = target_class.view(-1)

        # Calculate the intersection and union for this class
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        # Compute Dice coefficient for this class
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_score)

    return dice_scores


def dice_loss(pred, target, num_classes=4, smooth=1e-6):
    """
    Compute the Dice loss for multi-class segmentation by calculating Dice for each class separately.
    The total loss is the mean of Dice loss for each class.
    Args:
        pred: predicted tensor (logits or probabilities)
        target: ground truth tensor (long integers representing class labels)
        num_classes: number of classes (default is 4)
        smooth: small constant to avoid division by zero
    Returns:
        Mean Dice loss (1 - Dice coefficient for each class)
    """
    # Ensure pred is of float type for softmax
    pred = pred.float()

    # Apply softmax to convert logits to probabilities
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities

    # Get the Dice scores for each class
    dice_scores = dice_coeff(pred, target, num_classes, smooth)

    # Return the mean of the Dice loss (1 - Dice coefficient) for all classes
    return 1 - torch.mean(torch.tensor(dice_scores))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=4):  # 4 output channels for 4 classes
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Expanding path (Decoder)
        self.upconv4 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(128 + 128, 64)
        self.upconv2 = self.upconv_block(64 + 64, 32)
        self.upconv1 = self.upconv_block(32 + 32, 16)

        # Final layer to reduce channels before final convolution
        self.reduce_channels = nn.Conv3d(16 + 16, 16, kernel_size=1)

        # Final output layer for 4 classes
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # This upscales by a factor of 2
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding (downsampling)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoding (upsampling)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection

        # Reduce channels before final convolution
        dec1 = self.reduce_channels(dec1)

        # Final output layer to match the input size (128x128x128)
        output = self.final_conv(dec1)

        # Ensure the output is the same size as the target (128x128x128)
        output = F.interpolate(output, size=(128, 128, 128), mode='trilinear', align_corners=True)

        return output


class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        patient_id = image_file.split('_')[0]  # Assuming file name has patient ID (e.g., BraTS-GLI-02088)

        # Load the image modalities (T1c, T1n, T2f, T2w)
        t1c = nib.load(os.path.join(self.image_dir, f'{patient_id}_t1c.nii.gz')).get_fdata()
        t1n = nib.load(os.path.join(self.image_dir, f'{patient_id}_t1n.nii.gz')).get_fdata()
        t2f = nib.load(os.path.join(self.image_dir, f'{patient_id}_t2f.nii.gz')).get_fdata()
        t2w = nib.load(os.path.join(self.image_dir, f'{patient_id}_t2w.nii.gz')).get_fdata()

        # Load the mask
        mask = nib.load(os.path.join(self.mask_dir, f'{patient_id}_seg.nii.gz')).get_fdata()

        # Resize image and mask
        image = np.stack([t1c, t1n, t2f, t2w], axis=-1)  # Stack modalities into one image
        image = resize(image, (128, 128, 128, 4), mode='constant', preserve_range=True)  # Resize to fixed shape
        mask = resize(mask, (128, 128, 128), mode='constant', preserve_range=True)  # Resize mask

        # If transform is specified, apply it
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert to PyTorch tensors and return
        mask = torch.tensor(mask, dtype=torch.long)  # Mask should be of long dtype for multi-class segmentation
        return torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2), mask


# Define the model
model = UNet(in_channels=4, out_channels=4).to(device)  # 4 output channels for 4 classes

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Dataset and DataLoader
image_dir = '/home/ubuntu/DL-PROJECT/training_data1_v2/data/images'
mask_dir = '/home/ubuntu/DL-PROJECT/training_data1_v2/data/masks'
train_dataset = BraTSDataset(image_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

def check_for_nan(tensor, name="Tensor"):
    """Check if there are any NaN or Inf values in a tensor."""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values")
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf values")

# Training loop with progress tracking
num_epochs = 1
# Initialize the scaler for mixed precision
scaler = GradScaler()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    epoch_start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            outputs = model(inputs)
            loss = dice_loss(outputs, labels)  # Use Dice Loss

        # Check if the loss is NaN
        if torch.isnan(loss).any():
            print(f"Warning: NaN in loss at iteration {i}")
            continue  # Skip the current iteration if loss is NaN

        scaler.scale(loss).backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Calculate accuracy (for multi-class, we calculate accuracy for each class)
        preds = torch.argmax(outputs, dim=1)  # Get class predictions for each voxel
        correct_preds += (preds == labels).sum().item()
        total_preds += labels.numel()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    # Calculate and display accuracy and time for the epoch
    epoch_end_time = time.time()
    accuracy = correct_preds / total_preds * 100
    epoch_time = epoch_end_time - epoch_start_time

    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)

    avg_epoch_loss = running_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {minutes}m {seconds}s")

torch.save(model.state_dict(), 'unet_model.pt')
print("Model saved as 'unet_model.pt'")