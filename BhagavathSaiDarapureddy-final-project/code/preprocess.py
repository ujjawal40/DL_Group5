import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


#%%
def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)


#%%
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


#%%

class FCNN3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNN3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


#%%

def dice_coefficient(pred, target, num_classes):
    dice = 0.0
    for class_idx in range(1, num_classes):  # Exclude background
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice += (2 * intersection) / (union + 1e-5)
    return dice / (num_classes - 1)


def compute_hausdorff(pred, target):
    pred_np = (pred > 0).cpu().numpy()  # Binary prediction mask
    target_np = (target > 0).cpu().numpy()  # Binary ground truth mask

    pred_coords = np.argwhere(pred_np)
    target_coords = np.argwhere(target_np)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float('inf')

    h1 = directed_hausdorff(pred_coords, target_coords)[0]
    h2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(h1, h2)


def train_fcnn(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_directory, num_classes=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scaler = GradScaler()  # Initialize the GradScaler for mixed precision

    train_losses, val_losses = [], []
    train_dice_scores, val_dice_scores = [], []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss, total_train_dice = 0, 0
        total_batches = len(train_loader)

        for batch_idx, (images_original, masks_original, _, _) in enumerate(train_loader):
            images_original, masks_original = images_original.to(device), masks_original.to(device)

            optimizer.zero_grad()

            # Mixed precision training using autocast for the forward pass
            with autocast():  # Use autocast for mixed precision
                outputs = model(images_original)
                loss = criterion(outputs, masks_original)

            # Backpropagation using GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # Update the scaler after each step

            # Predictions and Dice score calculation
            preds = torch.argmax(outputs, dim=1)  # Get the index of the max log-probability
            dice_score = dice_coefficient(preds, masks_original, num_classes)
            total_train_dice += dice_score.item()

            total_train_loss += loss.item()

            # Debugging output for each batch
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, '
                  f'Loss: {loss.item():.4f}, Dice Score: {dice_score:.4f}')

        train_losses.append(total_train_loss / len(train_loader))
        train_dice_scores.append(total_train_dice / len(train_loader))

        # Validation phase
        model.eval()
        total_val_loss, total_val_dice = 0, 0
        with torch.no_grad():
            for batch_idx, (images_original, masks_original, _, _) in enumerate(val_loader):
                images_original, masks_original = images_original.to(device), masks_original.to(device)
                outputs = model(images_original)
                loss = criterion(outputs, masks_original)

                preds = torch.argmax(outputs, dim=1)
                dice = dice_coefficient(preds, masks_original, num_classes)
                total_val_loss += loss.item()
                total_val_dice += dice.item()

        val_losses.append(total_val_loss / len(val_loader))
        val_dice_scores.append(total_val_dice / len(val_loader))

        # Print metrics for the epoch
        print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, "
              f"Train Dice: {train_dice_scores[-1]:.4f}, "
              f"Validation Loss: {val_losses[-1]:.4f}, Validation Dice: {val_dice_scores[-1]:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth'))

    # Save training history to CSV
    history_df = pd.DataFrame({
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Dice': train_dice_scores,
        'Validation Dice': val_dice_scores,
    })
    history_df.to_csv(os.path.join(save_directory, 'training_history.csv'), index=False)
    print("Training complete. History saved.")
    return history_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image_dir = '/home/ubuntu/DL-PROJECT/combined_data/train_images'
train_mask_dir = '/home/ubuntu/DL-PROJECT/combined_data/masks'

val_img_dir = '/home/ubuntu/DL-PROJECT/val_data/val_images'
val_mask_dir = '/home/ubuntu/DL-PROJECT/val_data/val_masks'

save_dir='/home/ubuntu/dl_code/fcnn_models'
### Main Execution ###
print("Starting preprocessing...")
train_dataset = BraTSDataset(train_image_dir, train_mask_dir)
val_dataset = BraTSDataset(val_img_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
print("Preprocessing complete.")

model = FCNN3D(in_channels=4, out_channels=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = train_fcnn(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, save_directory=save_dir)

history_df = pd.read_csv(os.path.join(save_dir, 'training_history.csv'))
plt.figure(figsize=(12, 6))
plt.plot(history_df['Train Loss'], label='Train Loss')
plt.plot(history_df['Validation Loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history_df['Train Dice'], label='Train Dice')
plt.plot(history_df['Validation Dice'], label='Validation Dice')
plt.title('Dice Coefficient vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.show()
