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
