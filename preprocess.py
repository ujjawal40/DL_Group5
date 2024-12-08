import os
import SimpleITK as sitk
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the Code folder
dl_group5_dir = os.path.dirname(script_dir)  # Path to DL_Group5 folder
base_data_dir = os.path.join(dl_group5_dir, 'Data')  # Correct path to Data folder
data_dir = os.path.join(base_data_dir, 'training_data1_v2')  # Path to training_data1_v2

train_image_dir = os.path.join(data_dir, 'train_images')
train_mask_dir = os.path.join(data_dir, 'train_masks')


def load_image(image_path):
    """Load a NIfTI image using SimpleITK."""
    img = sitk.ReadImage(image_path)
    return img


def resample_image(image, target_spacing=[1.0, 1.0, 1.0], is_mask=False):
    """
    Resample the image to a uniform voxel spacing.
    Args:
        image (SimpleITK.Image): Input image.
        target_spacing (list): Desired voxel spacing (e.g., [1.0, 1.0, 1.0]).
        is_mask (bool): Whether the input image is a mask (use nearest neighbor interpolation).
    Returns:
        SimpleITK.Image: Resampled image.
    """
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


def create_brain_mask(image_data):
    """Create a binary brain mask."""
    return (image_data > 0).astype(np.uint8)


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

def crop_to_bounding_box(image_data, mask_data, padding=20):
    """
    Crop the image and mask to the bounding box of the brain/tumor region.
    Args:
        image_data (numpy.ndarray): 4D array (modalities, x, y, z).
        mask_data (numpy.ndarray): 3D array (binary mask of the brain region).
        padding (int): Number of voxels to pad around the bounding box.
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Cropped image and mask arrays.
    """
    coords = np.array(np.nonzero(mask_data))  # Find non-zero voxel coordinates in the mask
    min_coords = np.maximum(np.min(coords, axis=1) - padding, 0)  # Min coordinates with padding
    max_coords = np.minimum(np.max(coords, axis=1) + padding, mask_data.shape)  # Max coordinates with padding

    slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))

    # Crop image_data (4D) and mask_data (3D)
    cropped_image = image_data[:, slices[0], slices[1], slices[2]]  # Apply slices to spatial dimensions only
    cropped_mask = mask_data[slices]  # Apply slices to the mask

    return cropped_image, cropped_mask


def preprocess_subject(image_paths, mask_path, padding=20):
    """
    Preprocess modalities and their corresponding mask for a single subject.
    Includes resampling to uniform voxel spacing.
    """
    images = []
    brain_mask = None

    for path in image_paths:
        img = load_image(path)
        resampled_img = resample_image(img, target_spacing=[1.0, 1.0, 1.0])
        img_data = sitk.GetArrayFromImage(resampled_img)

        if brain_mask is None:
            brain_mask = create_brain_mask(img_data)

        clipped_image = clip_intensity(img_data)
        normalized_image = normalize_intensity(clipped_image)
        images.append(normalized_image)

    # Stack modalities
    stacked_images = np.stack(images, axis=0)

    # Process mask
    mask_img = load_image(mask_path)
    resampled_mask = resample_image(mask_img, target_spacing=[1.0, 1.0, 1.0], is_mask=True)
    mask_data = sitk.GetArrayFromImage(resampled_mask)

    # Crop to bounding box
    cropped_images, cropped_mask = crop_to_bounding_box(stacked_images, brain_mask, padding)

    return cropped_images, cropped_mask


def preprocess_dataset(image_dir, mask_dir, modalities, padding=20):

    """Preprocess an entire dataset."""

    subjects = sorted(set(f.split('_')[0] for f in os.listdir(image_dir) if f.endswith('.nii')))
    preprocessed_data = []
    # count = 1
    for subject in subjects:
        image_paths = [os.path.join(image_dir, f"{subject}_{mod}.nii") for mod in modalities]
        mask_path = os.path.join(mask_dir, f"{subject}_seg.nii")

        if all(os.path.exists(p) for p in image_paths) and os.path.exists(mask_path):
            preprocessed_images, preprocessed_mask = preprocess_subject(image_paths, mask_path, padding)
            # print(f"Appended {count} data point/s")
            # count += 1
            preprocessed_data.append((preprocessed_images, preprocessed_mask))

    return preprocessed_data


# Modalities
modalities = ['t1c', 't2f', 't2w']



# Preprocess training data
preprocessed_train_data = preprocess_dataset(train_image_dir, train_mask_dir, modalities)
print(f"Preprocessed {len(preprocessed_train_data)} training subjects.")
print("All Good, Proceed Further!. ")