#%% --------------------------------------- Imports --------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For reading .nii files
import numpy as np
import torch.nn.functional as F

#%% --------------------------------------- Data Prep ------------------------------------------------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()

    def _get_samples(self):
        samples = []
        for patient_folder in os.listdir(self.root_dir):
            modalities = []
            folder_path = os.path.join(self.root_dir, patient_folder)
            if os.path.isdir(folder_path):
                for filename in sorted(os.listdir(folder_path)):
                    if filename.endswith('.nii'):
                        file_path = os.path.join(folder_path, filename)
                        modalities.append(file_path)
                if len(modalities) == 5:
                    samples.append((modalities[:4], modalities[4]))  # (4 modalities, 1 segmentation mask)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        modality_paths, mask_path = self.samples[idx]

        # Load modalities as a 4D tensor (4, H, W, D)
        modalities = [nib.load(mod).get_fdata() for mod in modality_paths]
        x = np.stack(modalities)  # Shape: (4, 240, 240, 155)

        # Load mask and reshape to (1, H, W, D) for segmentation output
        y = nib.load(mask_path).get_fdata()
        y = y[np.newaxis, :, :, :]  # Shape: (1, 240, 240, 155)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)  # Assuming mask labels are integers

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

# Paths to training and validation data
train_data = BrainTumorDataset(root_dir='../Data/training_data1_v2')
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)


#%% --------------------------------------- Model Definiton ------------------------------------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Instantiate the model
model = UNet3D(in_channels=4, out_channels=2)  # Assuming 2 classes: tumor vs. non-tumor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
model = model.to(device)


#%% --------------------------------------- Model Training -------------------------------------------------------------
# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use cross-entropy for segmentation tasks
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(train_loader):

        inputs, masks = inputs.to(device), masks.to(device)

        # Forward pass
        outputs = model(inputs)
        # Assuming `outputs` is the model output and `masks` is the ground truth mask
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        masks = masks.squeeze(1)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


#%% --------------------------------------- Validation Loop ------------------------------------------------------------
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(val_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            # Reshape masks for compatibility
            masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
            masks = masks.squeeze(1)

            # Compute validation loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Paths to validation data
val_data = BrainTumorDataset(root_dir='../Data/validation_data1_v2')
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

#%% --------------------------------------- Training and Validation Integration ----------------------------------------
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)

        # Reshape masks
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        masks = masks.squeeze(1)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loss
    val_loss = validate_model(model, val_loader, criterion)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

#%% --------------------------------------- Model Testing --------------------------------------------------------------
def test_model(model, test_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
    return all_outputs

# Paths to testing data
test_data = BrainTumorDataset(root_dir='../Data/test_data1_v2')
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# Perform testing
test_outputs = test_model(model, test_loader)
print(f"Test completed on {len(test_outputs)} samples.")

#%% --------------------------------------- Dice Coefficient -----------------------------------------------------------
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.argmax(dim=1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

# Evaluate Dice score on validation set
dice_scores = []
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        dice = dice_coefficient(outputs, masks)
        dice_scores.append(dice)

avg_dice_score = sum(dice_scores) / len(dice_scores)
print(f"Average Dice Coefficient on Validation Set: {avg_dice_score}")

#%% --------------------------------------- Visualization --------------------------------------------------------------
import matplotlib.pyplot as plt

def visualize_sample(input_tensor, mask_tensor, pred_tensor):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display one slice from each tensor
    slice_idx = input_tensor.shape[-1] // 2

    ax[0].imshow(input_tensor[0, :, :, slice_idx], cmap='gray')
    ax[0].set_title('Input Slice')

    ax[1].imshow(mask_tensor[0, :, :, slice_idx], cmap='jet')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(pred_tensor[0, :, :, slice_idx], cmap='jet')
    ax[2].set_title('Prediction')

    plt.show()

# Visualize a random sample from validation set
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1, keepdim=True)
        visualize_sample(inputs.cpu().numpy()[0], masks.cpu().numpy(), outputs.cpu().numpy()[0])
        break

