import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import directed_hausdorff


# Define Paths
train_image_dir = '/home/ubuntu/DL_Group5/Data/train_images'
train_mask_dir = '/home/ubuntu/DL_Group5/Data/train_masks'
val_image_dir = '/home/ubuntu/DL_Group5/Data/test_images'
val_mask_dir = '/home/ubuntu/DL_Group5/Data/test_masks'
save_dir = '/home/ubuntu/DL_Group5/models/fcnn_models'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)


### Dataset and Preprocessing ###
class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(128, 128, 128)):
        print("Initializing dataset...")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.modalities = ['t1c', 't1n', 't2f', 't2w']
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii')])

    def __len__(self):
        return len(self.image_files) // len(self.modalities)

    def __getitem__(self, idx):
        patient_id = self.image_files[idx * len(self.modalities)].split('_')[0]
        images = []
        for modality in self.modalities:
            image_path = os.path.join(self.image_dir, f"{patient_id}_{modality}.nii")
            print(f"Loading and processing image: {image_path}")
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)
            image_resized = self.resize_image(image)
            image_np = sitk.GetArrayFromImage(image_resized)
            image_normalized = self.normalize_image(image_np)
            images.append(image_normalized)

        images_stack = np.stack(images, axis=0)

        mask_path = os.path.join(self.mask_dir, f"{patient_id}_seg.nii")
        print(f"Loading and processing mask: {mask_path}")
        mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask_resized = self.resize_image(mask)
        mask_np = sitk.GetArrayFromImage(mask_resized)

        return torch.tensor(images_stack, dtype=torch.float32), torch.tensor(mask_np, dtype=torch.long)

    def resize_image(self, image):
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([image.GetSpacing()[0] * (image.GetSize()[0] / self.target_size[0]),
                                    image.GetSpacing()[1] * (image.GetSize()[1] / self.target_size[1]),
                                    image.GetSpacing()[2] * (image.GetSize()[2] / self.target_size[2])])
        resampler.SetSize(self.target_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(image)

    def normalize_image(self, image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))


### FCNN Architecture ###
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


### Training and Validation ###
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
    scaler = GradScaler()

    train_losses, val_losses = [], []
    train_dice_scores, val_dice_scores = [], []
    train_hausdorff_scores, val_hausdorff_scores = [], []

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_train_loss, total_train_dice, total_train_hausdorff = 0, 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)
            dice = dice_coefficient(preds, masks, num_classes)
            hausdorff = compute_hausdorff(preds, masks)

            total_train_loss += loss.item()
            total_train_dice += dice.item()
            total_train_hausdorff += hausdorff

        train_losses.append(total_train_loss / len(train_loader))
        train_dice_scores.append(total_train_dice / len(train_loader))
        train_hausdorff_scores.append(total_train_hausdorff / len(train_loader))

        model.eval()
        total_val_loss, total_val_dice, total_val_hausdorff = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                preds = torch.argmax(outputs, dim=1)
                dice = dice_coefficient(preds, masks, num_classes)
                hausdorff = compute_hausdorff(preds, masks)

                total_val_loss += loss.item()
                total_val_dice += dice.item()
                total_val_hausdorff += hausdorff

        val_losses.append(total_val_loss / len(val_loader))
        val_dice_scores.append(total_val_dice / len(val_loader))
        val_hausdorff_scores.append(total_val_hausdorff / len(val_loader))

        print(f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
              f"Train Dice: {train_dice_scores[-1]:.4f}, Val Dice: {val_dice_scores[-1]:.4f}, "
              f"Train Hausdorff: {train_hausdorff_scores[-1]:.4f}, Val Hausdorff: {val_hausdorff_scores[-1]:.4f}")

        torch.save(model.state_dict(), os.path.join(save_directory, f'model_epoch_{epoch + 1}.pth'))

    history_df = pd.DataFrame({
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Dice': train_dice_scores,
        'Validation Dice': val_dice_scores,
        'Train Hausdorff': train_hausdorff_scores,
        'Validation Hausdorff': val_hausdorff_scores
    })
    history_df.to_csv(os.path.join(save_directory, 'training_history.csv'), index=False)
    print("Training complete. History saved.")
    return history_df


### Main Execution ###
print("Starting preprocessing...")
train_dataset = MedicalDataset(train_image_dir, train_mask_dir)
val_dataset = MedicalDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
print("Preprocessing complete.")

model = FCNN3D(in_channels=4, out_channels=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = train_fcnn(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, save_directory=save_dir)

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






