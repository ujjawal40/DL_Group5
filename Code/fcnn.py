import os
import random
import nibabel as nib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from tqdm import tqdm
import gc
from torchvision import models
from collections import defaultdict
import copy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Seeding
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("> SEEDING DONE")


# Dataset Splitting
data_dir = '/home/ubuntu/DL_Group5/training_data1_v2'

# Read and shuffle subdirectories
sub_directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
random.shuffle(sub_directories)

# Split into train (80%) and validation (20%)
num_train = int(0.8 * len(sub_directories))
train_directories = sub_directories[:num_train]
validation_directories = sub_directories[num_train:]

print(f"Total directories: {len(sub_directories)}")
print(f"Training directories: {len(train_directories)}")
print(f"Validation directories: {len(validation_directories)}")


# Dataset Class
class BuildDataset(Dataset):
    def __init__(self, directories, data_dir, subset="train", transforms=None):
        self.directories = directories
        self.data_dir = data_dir
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        # Get the directory for this sample
        patient_dir = self.directories[index]
        patient_path = os.path.join(self.data_dir, patient_dir)

        # Define file paths
        t1c_file = os.path.join(patient_path, f"{patient_dir}-t1c.nii.gz")
        seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")

        # Load the image and mask
        img = nib.load(t1c_file).get_fdata()
        mask = nib.load(seg_file).get_fdata()

        # Resize to match model input size
        img = cv2.resize(img, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        # Normalize the image
        img = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0

        # Convert mask to one-hot encoding (3 classes)
        masks = np.zeros((128, 128, 3))
        for i in range(3):
            masks[:, :, i] = (mask == i).astype(np.float32)

        # Apply transformations (if any)
        if self.transforms:
            augmented = self.transforms(image=img, mask=masks)
            img = augmented["image"]
            masks = augmented["mask"]

        return torch.tensor(img).permute(2, 0, 1), torch.tensor(masks).permute(2, 0, 1)


# Read Data
def read_data():
    train_dataset = BuildDataset(train_directories, data_dir, subset="train")
    val_dataset = BuildDataset(validation_directories, data_dir, subset="val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    return train_loader, val_loader


# Visualization
def plot_batch(imgs, msks=None, size=3):
    size = min(size, len(imgs))  # Ensure size does not exceed batch size
    plt.figure(figsize=(5 * size, 5))
    for idx in range(size):
        plt.subplot(2, size, idx + 1)
        img = imgs[idx].permute((1, 2, 0)).cpu().numpy()
        plt.imshow(img, cmap="bone")
        plt.axis("off")
        plt.title("Image")

        if msks is not None:
            plt.subplot(2, size, idx + 1 + size)
            mask = msks[idx].permute((1, 2, 0)).cpu().numpy()
            plt.imshow(mask, cmap="viridis", alpha=0.5)
            plt.axis("off")
            plt.title("Mask")
    plt.tight_layout()
    plt.show()


# Model Definition
class FCNVGG(nn.Module):
    def __init__(self):
        super(FCNVGG, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.vgg_features = vgg16.features

        for param in self.vgg_features.parameters():
            param.requires_grad = False

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=7, padding=3)
        nn.init.kaiming_normal_(self.trans_conv1.weight)
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv2.weight)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv3.weight)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv4.weight)
        self.trans_conv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv7.weight)

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.conv5(x)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.conv6(x)
        return x


# Model Initialization
def model_definition():
    model = FCNVGG().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    return model, optimizer, criterion, scheduler


# Training Function
def train_one_epoch(model, optimizer, dataloader, criterion):
    model.train()
    running_loss = 0.0

    for imgs, msks in tqdm(dataloader, total=len(dataloader)):
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, msks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# Validation Function
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, msks in tqdm(dataloader, total=len(dataloader)):
            imgs, msks = imgs.to(device), msks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, msks)

            running_loss += loss.item()

    return running_loss / len(dataloader)


# Main Function
if __name__ == '__main__':
    set_seed()

    # Data loaders
    train_loader, val_loader = read_data()

    # Model and training setup
    model, optimizer, criterion, scheduler = model_definition()

    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        train_loss = train_one_epoch(model, optimizer, train_loader, criterion)
        val_loss = validate_one_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "model_fcn_vgg.pth")
