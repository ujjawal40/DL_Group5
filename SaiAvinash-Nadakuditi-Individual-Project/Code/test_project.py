import os
import nibabel as nib
import numpy as np

def load_image(image_path):
    """Load a NIfTI image and return its data as a NumPy array."""
    img = nib.load(image_path)
    return img.get_fdata(), img.affine, img.header

def create_brain_mask(image_data):
    return (image_data > 0).astype(np.uint8)

def clip_intensity(image_data, lower_percentile=2.5, upper_percentile=97.5):
    lower = np.percentile(image_data, lower_percentile)
    upper = np.percentile(image_data, upper_percentile)
    return np.clip(image_data, lower, upper)

def normalize_intensity(image_data):
    mean = np.mean(image_data)
    std = np.std(image_data)
    return (image_data - mean) / (std + 1e-8)

def crop_to_bounding_box(image_data, mask_data, padding=20):
    coords = np.array(np.nonzero(mask_data))
    min_coords = np.maximum(np.min(coords, axis=1) - padding, 0)
    max_coords = np.minimum(np.max(coords, axis=1) + padding, image_data.shape)
    slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    return image_data[slices], mask_data[slices]

def preprocess_subject(image_paths, mask_path, padding=20):
    images = []
    brain_mask = None

    # Process each modality
    for path in image_paths:
        img_data, _, _ = load_image(path)

        if brain_mask is None:
            brain_mask = create_brain_mask(img_data)

        clipped_image = clip_intensity(img_data)
        normalized_image = normalize_intensity(clipped_image)
        images.append(normalized_image)

    # Stack modalities
    stacked_images = np.stack(images, axis=0)

    # Process the mask
    mask_data, _, _ = load_image(mask_path)
    cropped_images, cropped_mask = crop_to_bounding_box(stacked_images, brain_mask, padding)

    return cropped_images, cropped_mask

def preprocess_dataset(image_dir, mask_dir, modalities, padding=20):
    """
    Preprocess an entire dataset, preparing it for augmentation.
    Args:
        image_dir (str): Directory containing input images.
        mask_dir (str): Directory containing segmentation masks.
        modalities (list): List of modality suffixes (e.g., ['t1c', 't2f', 'flair']).
        padding (int): Padding around the bounding box during cropping.
    Returns:
        list: List of tuples (preprocessed_images, preprocessed_mask) for all subjects.
    """
    subjects = sorted(set(f.split('_')[0] for f in os.listdir(image_dir)))
    preprocessed_data = []

    for subject in subjects:
        image_paths = [os.path.join(image_dir, f"{subject}_{mod}.nii") for mod in modalities]
        mask_path = os.path.join(mask_dir, f"{subject}_seg.nii")

        preprocessed_images, preprocessed_mask = preprocess_subject(image_paths, mask_path, padding)
        preprocessed_data.append((preprocessed_images, preprocessed_mask))

    return preprocessed_data

# Directories for train
test_image_dir = '/path/to/test_images'
test_mask_dir = '/path/to/test_masks'

# Modalities
modalities = ['t1c', 't2f', 't2w']

# Preprocess testing data (if needed)
preprocessed_test_data = preprocess_dataset(test_image_dir, test_mask_dir, modalities)
print(f"Preprocessed {len(preprocessed_test_data)} testing subjects.")
