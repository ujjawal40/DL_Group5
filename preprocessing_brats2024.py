import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def load_image(path):
    """Load an image using SimpleITK and ensure it is in float32 format."""
    return sitk.ReadImage(path, sitk.sitkFloat32)


def apply_transformations(image):
    """Apply zooming, cropping, and rotation transformations to the image using SimpleITK."""

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


def normalize_image(image_np):
    """Normalize image array to have pixel values between 0 and 1."""
    image_min = np.min(image_np)
    image_max = np.max(image_np)
    if image_max > image_min:
        image_np = (image_np - image_min) / (image_max - image_min)
    return image_np

class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(155, 240, 240,)):
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
        images = []
        for mod in self.modalities:
            mod_path = os.path.join(self.image_dir, f'{patient_id}_{mod}.nii')
            image = load_image(mod_path)
            image = apply_transformations(image)
            image = resize_image(image, self.target_size)
            image_np = sitk.GetArrayFromImage(image)
            normalized_image = normalize_image(image_np)
            images.append(normalized_image)

        image_stack = np.stack(images, axis=-1)
        seg_path = os.path.join(self.mask_dir, f'{patient_id}_seg.nii')
        segmentation = load_image(seg_path)
        segmentation = apply_transformations(segmentation)
        segmentation = resize_image(segmentation, self.target_size)
        seg_array = sitk.GetArrayFromImage(segmentation)

        return torch.tensor(image_stack, dtype=torch.float32), torch.tensor(seg_array, dtype=torch.long)
def custom_collate(batch):
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

def visualize_batch(images, segmentations):
    batch_size = images.shape[0]
    print(images.shape)
    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 6 * batch_size))
    for i in range(batch_size):
        axes[i, 0].imshow(images[i, :, :, images.shape[2] // 2, 2], cmap='gray')
        axes[i, 0].set_title('Transformed and Normalized Image (t1c)')
        axes[i, 1].imshow(segmentations[i, :, :, segmentations.shape[2] // 2], cmap='gray')
        axes[i, 1].set_title('Transformed Segmentation Mask')
    plt.show()

img_dir = '/home/ubuntu/Dataset/combined_data/train_images'
mask_dir = '/home/ubuntu/Dataset/combined_data/masks'

dataset = BraTSDataset(img_dir, mask_dir)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate)

for images, segmentations in data_loader:
    visualize_batch(images, segmentations)
    break  # Comment out or remove this line to visualize more batches
