import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random
import os

def augment_image_and_mask(image, segmentation):
    """Apply random augmentations to both image and segmentation."""
    # Random Zoom
    if random.random() > 0.5:
        zoom_factor = random.uniform(1.0, 1.3)  # Random zoom between 1.0 and 1.3
        size = image.GetSize()
        spacing = image.GetSpacing()
        new_spacing = [s / zoom_factor for s in spacing]
        resampler_zoom = sitk.ResampleImageFilter()
        resampler_zoom.SetOutputSpacing(new_spacing)
        resampler_zoom.SetSize([int(size[i] * zoom_factor) for i in range(3)])
        resampler_zoom.SetInterpolator(sitk.sitkLinear)
        resampler_zoom.SetReferenceImage(image)
        image = resampler_zoom.Execute(image)
        segmentation = resampler_zoom.Execute(segmentation)

    # Random Crop
    if random.random() > 0.5:
        crop_size = [int(0.8 * s) for s in image.GetSize()]  # Crop to 80% size
        crop_start = [random.randint(0, image.GetSize()[i] - crop_size[i]) for i in range(3)]
        crop_filter = sitk.RegionOfInterestImageFilter()
        crop_filter.SetSize(crop_size)
        crop_filter.SetIndex(crop_start)
        image = crop_filter.Execute(image)
        segmentation = crop_filter.Execute(segmentation)

    # Random Flip
    if random.random() > 0.5:
        flip_axes = random.choice([0, 1, 2])  # Choose a random axis
        image = sitk.Flip(image, [flip_axes == i for i in range(3)])
        segmentation = sitk.Flip(segmentation, [flip_axes == i for i in range(3)])

    # Random Rotation
    if random.random() > 0.5:
        angle = random.uniform(-np.pi / 12, np.pi / 12)  # Random rotation between -15° and 15°
        transform = sitk.AffineTransform(3)
        transform.Rotate(0, 1, angle)
        resampler_rotate = sitk.ResampleImageFilter()
        resampler_rotate.SetTransform(transform)
        resampler_rotate.SetInterpolator(sitk.sitkLinear)
        resampler_rotate.SetReferenceImage(image)
        image = resampler_rotate.Execute(image)
        segmentation = resampler_rotate.Execute(segmentation)

    return image, segmentation

def apply_transformations_all(image):
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
    transformations["zoomed"] = zoomed_image

    # ---- Cropping ----
    crop_size = [int(0.8 * s) for s in zoomed_image.GetSize()]  # Crop to 80% of the zoomed image size
    crop_start = [(zoomed_image.GetSize()[i] - crop_size[i]) // 2 for i in range(3)]  # Center cropping
    cropped_image = sitk.RegionOfInterest(zoomed_image, crop_size, crop_start)
    transformations["cropped"] = cropped_image

    # ---- Rotation ----
    transform = sitk.AffineTransform(3)  # 3D affine transformation
    transform.Rotate(0, 1, np.pi / 12)  # Rotate around the z-axis
    resampler_rotate = sitk.ResampleImageFilter()
    resampler_rotate.SetInterpolator(sitk.sitkLinear)
    resampler_rotate.SetTransform(transform)
    resampler_rotate.SetReferenceImage(cropped_image)
    rotated_image = resampler_rotate.Execute(cropped_image)
    transformations["rotated"] = rotated_image

    return transformations

def visualize_augmented(image, segmentation, augmented_image, augmented_segmentation):
    """Visualize original and augmented images along with their segmentations."""
    image_np = sitk.GetArrayFromImage(image)
    segmentation_np = sitk.GetArrayFromImage(segmentation)
    augmented_image_np = sitk.GetArrayFromImage(augmented_image)
    augmented_segmentation_np = sitk.GetArrayFromImage(augmented_segmentation)

    slice_idx = image_np.shape[0] // 2  # Middle slice for visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image and segmentation
    axes[0, 0].imshow(image_np[slice_idx], cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 1].imshow(segmentation_np[slice_idx], cmap='gray')
    axes[0, 1].set_title("Original Segmentation")

    # Augmented image and segmentation
    axes[1, 0].imshow(augmented_image_np[slice_idx], cmap='gray')
    axes[1, 0].set_title("Augmented Image")
    axes[1, 1].imshow(augmented_segmentation_np[slice_idx], cmap='gray')
    axes[1, 1].set_title("Augmented Segmentation")

    plt.show()

def visualize_label(segmentation, label=2):
    """
    Visualize only the specified label in the segmentation image.
    Args:
        segmentation: The segmentation SimpleITK image.
        label: The label to visualize (default: 2).
    """
    segmentation_np = sitk.GetArrayFromImage(segmentation)

    # Filter the segmentation to show only the specified label
    label_only = (segmentation_np == label).astype(np.uint8)

    slice_idx = label_only.shape[0] // 2  # Middle slice for visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(label_only[slice_idx], cmap='gray')
    plt.title(f"Segmentation (Label {label})")
    plt.axis('off')
    plt.show()

def visualize_transformations(transformations):
    """
    Visualize zoomed, cropped, and rotated images.
    Args:
        transformations: Dictionary with keys ['zoomed', 'cropped', 'rotated'].
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    slice_idx = transformations["zoomed"].GetSize()[2] // 2  # Middle slice

    for i, (key, img) in enumerate(transformations.items()):
        img_np = sitk.GetArrayFromImage(img)
        axes[i].imshow(img_np[slice_idx], cmap='gray')
        axes[i].set_title(key.capitalize())
        axes[i].axis('off')

    plt.show()

# Example Usage

img_dir = '/home/ubuntu/Dataset/combined_data/train_images'
mask_dir = '/home/ubuntu/Dataset/combined_data/masks'
images=os.listdir(img_dir)
image_name=images[11]
patient_id = image_name.split('_')[0]
mod='t1c'
seg='seg'
image_path =  os.path.join(img_dir, f'{patient_id}_{mod}.nii')  # Replace with actual path
segmentation_path = os.path.join(mask_dir, f'{patient_id}_{seg}.nii') # Replace with actual path

image = sitk.ReadImage(image_path, sitk.sitkFloat32)
segmentation = sitk.ReadImage(segmentation_path, sitk.sitkUInt8)
#
# transformations = apply_transformations_all(image)
#
# # Visualize the intermediate transformations
# visualize_transformations(transformations)
#
# Apply augmentations
augmented_image, augmented_segmentation = augment_image_and_mask(image, segmentation)

# Visualize original and augmented data
visualize_augmented(image, segmentation, augmented_image, augmented_segmentation)
visualize_label(segmentation, label=1)
