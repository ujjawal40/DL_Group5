import os
import SimpleITK as sitk
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dl_group5_dir = os.path.dirname(script_dir)
data_dir = os.path.join(dl_group5_dir, 'nData')

train_image_dir = os.path.join(data_dir, 'train_images')
train_mask_dir = os.path.join(data_dir, 'train_masks')

modalities = ['t1c', 't2f', 't2w', 't1n']

# Utility Functions
def load_image(image_path):
    """Load a NIfTI image using SimpleITK."""
    img = sitk.ReadImage(image_path)
    return img

def resample_image(image, target_spacing=[1.0, 1.0, 1.0], is_mask=False):
    """Resample the image to a uniform voxel spacing."""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)

def clip_intensity(image_data, lower_percentile=2.5, upper_percentile=97.5):
    """Clip intensity values to specified percentiles."""
    lower = np.percentile(image_data, lower_percentile)
    upper = np.percentile(image_data, upper_percentile)
    return np.clip(image_data, lower, upper)

def normalize_intensity(image_data):
    """Normalize intensity values to have zero mean and unit variance."""
    mean = np.mean(image_data)
    std = np.std(image_data)
    return (image_data - mean) / (std + 1e-8)

def augment_image(image, mask=None):
    """Apply data augmentation techniques."""
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
        if mask is not None:
            mask = np.flip(mask, axis=2)

    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        if mask is not None:
            mask = np.flip(mask, axis=1)

    angle = np.radians(random.uniform(-20, 20))
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    if random.random() < 0.5:
        image = affine_transform(image, rotation_matrix)
        if mask is not None:
            mask = affine_transform(mask, rotation_matrix, order=0)

    alpha = 720
    sigma = 24
    if random.random() < 0.5:
        shape = image.shape[1:]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        indices = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        indices = [
            np.reshape(e + de, (-1,)) for e, de in zip(indices, [dx, dy, dz])
        ]
        image = map_coordinates(image, indices, order=1).reshape(shape)
        if mask is not None:
            mask = map_coordinates(mask, indices, order=0).reshape(shape)

    return image, mask

def preprocess_subject(image_paths, mask_path, padding=20):
    """Preprocess modalities and their corresponding mask for a single subject."""
    images = []
    brain_mask = None

    for path in image_paths:
        img = load_image(path)
        resampled_img = resample_image(img, target_spacing=[1.0, 1.0, 1.0])
        img_data = sitk.GetArrayFromImage(resampled_img)

        if brain_mask is None:
            brain_mask = (img_data > 0).astype(np.uint8)

        clipped_image = clip_intensity(img_data)
        normalized_image = normalize_intensity(clipped_image)
        images.append(normalized_image)

    stacked_images = np.stack(images, axis=0)

    mask_img = load_image(mask_path)
    resampled_mask = resample_image(mask_img, target_spacing=[1.0, 1.0, 1.0], is_mask=True)
    mask_data = sitk.GetArrayFromImage(resampled_mask)

    augmented_images, augmented_mask = augment_image(stacked_images, mask_data)

    return augmented_images, augmented_mask

def preprocess_dataset_generator(image_dir, mask_dir, modalities, padding=20):
    """Preprocess an entire dataset and yield data directly."""
    subjects = sorted(set(f.split('_')[0] for f in os.listdir(image_dir) if f.endswith('.nii')))
    for subject in subjects:
        image_paths = [os.path.join(image_dir, f"{subject}_{mod}.nii") for mod in modalities]
        mask_path = os.path.join(mask_dir, f"{subject}_seg.nii")

        if all(os.path.exists(p) for p in image_paths) and os.path.exists(mask_path):
            preprocessed_images, preprocessed_mask = preprocess_subject(image_paths, mask_path, padding)
            yield preprocessed_images, preprocessed_mask

# Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

# Define Generator and Discriminator (same as earlier)
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

# Loss Functions
adversarial_loss = nn.BCELoss()
segmentation_loss = nn.CrossEntropyLoss()

# Evaluation Metrics
def compute_dice(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def compute_hausdorff(pred, target):
    pred = (pred > 0.5).cpu().numpy()
    target = target.cpu().numpy()
    pred_coords = np.argwhere(pred)
    target_coords = np.argwhere(target)
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return float('inf')
    h1 = directed_hausdorff(pred_coords, target_coords)[0]
    h2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(h1, h2)

# Training Loop with Saving Parameters and Plotting
def train_gan_segnet(generator, discriminator, dataloader, num_epochs=50, lr=0.0002):
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
        generator.train()
        discriminator.train()
        epoch_g_loss = 0
        epoch_d_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Train discriminator
            optimizer_d.zero_grad()
            real_labels = torch.ones(images.size(0), 1).to(device)
            fake_labels = torch.zeros(images.size(0), 1).to(device)

            real_inputs = torch.cat((images, masks.unsqueeze(1)), dim=1)
            real_outputs = discriminator(real_inputs)
            d_loss_real = adversarial_loss(real_outputs, real_labels)

            fake_masks = generator(images)
            fake_inputs = torch.cat((images, fake_masks.detach()), dim=1)
            fake_outputs = discriminator(fake_inputs)
            d_loss_fake = adversarial_loss(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            epoch_d_loss += d_loss.item()

            # Train generator
            optimizer_g.zero_grad()
            fake_masks = generator(images)
            fake_inputs = torch.cat((images, fake_masks), dim=1)
            fake_outputs = discriminator(fake_inputs)

            g_loss_adv = adversarial_loss(fake_outputs, real_labels)
            g_loss_seg = segmentation_loss(fake_masks, masks)
            g_loss = g_loss_adv + g_loss_seg

            g_loss.backward()
            optimizer_g.step()
            epoch_g_loss += g_loss.item()

        g_losses_epoch.append(epoch_g_loss / len(dataloader))
        d_losses_epoch.append(epoch_d_loss / len(dataloader))

        # Evaluate metrics
        generator.eval()
        dice_scores = []
        hausdorff_distances = []
        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(device), masks.to(device)
                preds = generator(images)
                dice_scores.append(compute_dice(preds, masks))
                hausdorff_distances.append(compute_hausdorff(preds, masks))

        avg_dice = np.mean(dice_scores)
        avg_hausdorff = np.mean(hausdorff_distances)
        dice_scores_epoch.append(avg_dice)
        hausdorff_scores_epoch.append(avg_hausdorff)

        # Print and Save Metrics
        print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_losses_epoch[-1]:.4f}, G Loss: {g_losses_epoch[-1]:.4f}, "
              f"Dice: {avg_dice:.4f}, Hausdorff: {avg_hausdorff:.4f}")

        # Save model parameters
        torch.save(generator.state_dict(), os.path.join(script_dir, f"generator_epoch_{epoch + 1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(script_dir, f"discriminator_epoch_{epoch + 1}.pth"))

    # Plot metrics
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, g_losses_epoch, label='Generator Loss')
    plt.plot(epochs, d_losses_epoch, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, dice_scores_epoch, label='Dice Coefficient')
    plt.plot(epochs, hausdorff_scores_epoch, label='Hausdorff Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.show()

# Load Preprocessed Data
train_images = []
train_masks = []
for images, masks in preprocess_dataset_generator(train_image_dir, train_mask_dir, ['t1c', 't2f', 't2w', 't1n']):
    train_images.append(images)
    train_masks.append(masks)

# Dataset and DataLoader
train_dataset = BrainTumorDataset(train_images, train_masks)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Initialize Networks
generator = Generator(input_channels=4)
discriminator = Discriminator(input_channels=5)

# Train GAN-segNet
train_gan_segnet(generator, discriminator, train_loader)


