import os
import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import directed_hausdorff

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dl_group5_dir = os.path.dirname(script_dir)
data_dir = os.path.join(dl_group5_dir, 'Data')

train_image_dir = os.path.join(data_dir, 'train_images')
train_mask_dir = os.path.join(data_dir, 'train_masks')

modalities = ['t1c', 't2f', 't2w', 't1n']

# Utility Functions
def load_image(path):
    print(f"Loading image from {path}...")
    return sitk.ReadImage(path, sitk.sitkFloat32)

def resize_image(image, new_size):
    print(f"Resizing image to {new_size}...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing([image.GetSpacing()[i] * (image.GetSize()[i] / new_size[i]) for i in range(3)])
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    return resampler.Execute(image)

def normalize_image(image_np):
    print("Normalizing image...")
    image_min, image_max = np.min(image_np), np.max(image_np)
    return (image_np - image_min) / (image_max - image_min) if image_max > image_min else image_np

# Augmentation: Elastic Transform
def elastic_transform(image, alpha, sigma):
    print("Applying elastic transformation...")
    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = [np.clip(x + dx, 0, shape[0] - 1), np.clip(y + dy, 0, shape[1] - 1), np.clip(z + dz, 0, shape[2] - 1)]
    return map_coordinates(image, indices, order=1, mode='reflect')

# Preprocessing Function
def preprocess_subject(image_paths, mask_path, target_size=(132, 116, 132), augment=False):
    print(f"Preprocessing subject with images {image_paths} and mask {mask_path}...")
    images = []
    for path in image_paths:
        image = load_image(path)
        resized_image = resize_image(image, target_size)
        normalized_image = normalize_image(sitk.GetArrayFromImage(resized_image))
        images.append(normalized_image)

    stacked_images = np.stack(images, axis=0)

    mask = load_image(mask_path)
    resized_mask = resize_image(mask, target_size)
    mask_data = sitk.GetArrayFromImage(resized_mask)

    if augment:
        print("Applying augmentations...")
        alpha, sigma = 720, 24
        for i in range(stacked_images.shape[0]):
            stacked_images[i] = elastic_transform(stacked_images[i], alpha, sigma)
        mask_data = elastic_transform(mask_data, alpha, sigma)

    return torch.tensor(stacked_images, dtype=torch.float32), torch.tensor(mask_data, dtype=torch.long)

# Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(128, 128, 128), augment=False):
        print("Initializing dataset...")
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        self.modalities = modalities
        self.image_files = sorted([f for f in os.listdir(image_dir) if any(mod in f for mod in modalities) and f.endswith('.nii')])

    def __len__(self):
        return len(self.image_files) // len(self.modalities)

    def __getitem__(self, idx):
        patient_id = self.image_files[idx * len(self.modalities)].split('_')[0]
        image_paths = [os.path.join(self.image_dir, f"{patient_id}_{mod}.nii") for mod in self.modalities]
        mask_path = os.path.join(self.mask_dir, f"{patient_id}_seg.nii")
        return preprocess_subject(image_paths, mask_path, self.target_size, self.augment)

# Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, input_channels=4, output_channels=1):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Discriminator(nn.Module):
    def __init__(self, input_channels=5):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Add Evaluation Metrics
def compute_dice(pred, target):
    """Compute Dice Coefficient."""
    smooth = 1e-5  # To avoid division by zero
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def compute_hausdorff(pred, target):
    """Compute Hausdorff Distance."""
    pred = (pred > 0.5).cpu().numpy()
    target = target.cpu().numpy()
    pred_coords = np.argwhere(pred)
    target_coords = np.argwhere(target)

    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float("inf")  # Handle empty predictions or targets
    h1 = directed_hausdorff(pred_coords, target_coords)[0]
    h2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(h1, h2)


# Updated Training Loop with Metrics
def train_gan_segnet(generator, discriminator, train_loader, num_epochs=50, lr=0.0002):
    print("Training Started!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    # Store metrics for plotting
    dice_scores_epoch = []
    hausdorff_scores_epoch = []
    g_losses_epoch = []
    d_losses_epoch = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}...")
        generator.train()
        discriminator.train()
        epoch_g_loss = 0
        epoch_d_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            real_inputs = torch.cat((images, masks.unsqueeze(1)), dim=1)
            fake_masks = generator(images)
            fake_inputs = torch.cat((images, fake_masks.detach()), dim=1)

            d_loss_real = nn.BCELoss()(discriminator(real_inputs), real_labels)
            d_loss_fake = nn.BCELoss()(discriminator(fake_inputs), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            epoch_d_loss += d_loss.item()

            # Train generator
            optimizer_g.zero_grad()
            g_loss_adv = nn.BCELoss()(discriminator(fake_inputs), real_labels)
            g_loss_seg = nn.CrossEntropyLoss()(fake_masks, masks)
            g_loss = g_loss_adv + g_loss_seg
            g_loss.backward()
            optimizer_g.step()
            epoch_g_loss += g_loss.item()

        g_losses_epoch.append(epoch_g_loss / len(train_loader))
        d_losses_epoch.append(epoch_d_loss / len(train_loader))

        # Evaluate Metrics
        generator.eval()
        dice_scores = []
        hausdorff_distances = []
        with torch.no_grad():
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                preds = generator(images)
                dice_scores.append(compute_dice(preds, masks))
                hausdorff_distances.append(compute_hausdorff(preds, masks))

        avg_dice = np.mean(dice_scores)
        avg_hausdorff = np.mean(hausdorff_distances)
        dice_scores_epoch.append(avg_dice)
        hausdorff_scores_epoch.append(avg_hausdorff)

        # Print Metrics for the Current Epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_losses_epoch[-1]:.4f}, G Loss: {g_losses_epoch[-1]:.4f}, "
              f"Dice: {avg_dice:.4f}, Hausdorff: {avg_hausdorff:.4f}")

        # Save Model Parameters
        torch.save(generator.state_dict(), os.path.join(script_dir, f"generator_epoch_{epoch + 1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(script_dir, f"discriminator_epoch_{epoch + 1}.pth"))

    # Plot Metrics
    epochs = range(1, num_epochs + 1)

    # Plot Losses
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, g_losses_epoch, label="Generator Loss")
    plt.plot(epochs, d_losses_epoch, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.show()

    # Plot Dice Coefficient and Hausdorff Distance
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, dice_scores_epoch, label="Dice Coefficient")
    plt.plot(epochs, hausdorff_scores_epoch, label="Hausdorff Distance")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Evaluation Metrics")
    plt.show()


# Initialize Dataset and Dataloader
print("Loading dataset...")
train_dataset = BrainTumorDataset(train_image_dir, train_mask_dir, augment=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print("Dataset loaded successfully.")

# Initialize Models
print("Initializing models...")
generator = Generator(input_channels=4)
discriminator = Discriminator(input_channels=5)

# Train Model
train_gan_segnet(generator, discriminator, train_loader)


# Training Loop with Metrics Plotting
def train_gan_segnet(generator, discriminator, train_loader, num_epochs=50, lr=0.0002):
    print("Training Started!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    dice_scores_epoch, hausdorff_scores_epoch, g_losses_epoch, d_losses_epoch = [], [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}...")
        generator.train()
        discriminator.train()
        epoch_g_loss, epoch_d_loss = 0, 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            real_inputs = torch.cat((images, masks.unsqueeze(1)), dim=1)
            fake_masks = generator(images)
            fake_inputs = torch.cat((images, fake_masks.detach()), dim=1)

            d_loss_real = nn.BCELoss()(discriminator(real_inputs), real_labels)
            d_loss_fake = nn.BCELoss()(discriminator(fake_inputs), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            epoch_d_loss += d_loss.item()

            # Train generator
            optimizer_g.zero_grad()
            g_loss_adv = nn.BCELoss()(discriminator(fake_inputs), real_labels)
            g_loss_seg = nn.CrossEntropyLoss()(fake_masks, masks)
            g_loss = g_loss_adv + g_loss_seg
            g_loss.backward()
            optimizer_g.step()
            epoch_g_loss += g_loss.item()

        d_losses_epoch.append(epoch_d_loss / len(train_loader))
        g_losses_epoch.append(epoch_g_loss / len(train_loader))

        # Print metrics
        print(f"D Loss: {epoch_d_loss:.4f}, G Loss: {epoch_g_loss:.4f}")

    # Plotting Losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs + 1), d_losses_epoch, label="Discriminator Loss")
    plt.plot(range(1, num_epochs + 1), g_losses_epoch, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.show()

# Load Dataset
print("Loading dataset...")
train_dataset = BrainTumorDataset(train_image_dir, train_mask_dir, augment=True)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print("Dataset loaded successfully.")

# Initialize Models
print("Initializing models...")
generator = Generator(input_channels=4)
discriminator = Discriminator(input_channels=5)

# Train Model
train_gan_segnet(generator, discriminator, train_loader)


class Discriminator(nn.Module):
    def __init__(self, input_channels=5):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 1, kernel_size=3, padding=1),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Reduce to (batch_size, 1, 1, 1, 1)
        self.flatten = nn.Flatten()  # Final shape (batch_size, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return x

