import nibabel as nib
import numpy as np
import os
import re
import random


random.seed(4001)

# Get the directory in which the current script is located
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Navigate up one level to the parent folder, which is DL-PROJECT
parent_dir = os.path.dirname(current_script_path)

# Path to the training_data1_v2 folder
data_dir = os.path.join(parent_dir, 'training_data1_v2')

# print("Data directory:", data_dir)

sub_directories= os.listdir(data_dir)

l=len(sub_directories)


# Shuffle the list of subdirectories
random.shuffle(sub_directories)

# Calculate the number of folders for training (90%)
num_train = int(0.8 * len(sub_directories))

# Split the directories into training and validation sets
train_directories = sub_directories[:num_train]
validation_directories = sub_directories[num_train:]

print(f"Total directories: {len(sub_directories)}")
print(f"Training directories: {len(train_directories)}")  # Should be 90% of total
print(f"Validation directories: {len(validation_directories)}")



def brats_2024():
    brats_data_dir_2024 = data_dir

    train_images_dir = os.path.join(parent_dir, 'train_data', 'train_images')
    train_masks_dir = os.path.join(parent_dir, 'train_data', 'train_masks')

    # Ensure the output directories exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)

    for patient_dir in train_directories:
        patient_path = os.path.join(brats_data_dir_2024, patient_dir)

        # Check if it's a directory and follows the naming convention
        if os.path.isdir(patient_path) and patient_dir.startswith('BraTS-GLI'):
            # Define the files for this patient
            t1c_file = os.path.join(patient_path, f"{patient_dir}-t1c.nii.gz")
            t1n_file = os.path.join(patient_path, f"{patient_dir}-t1n.nii.gz")
            t2f_file = os.path.join(patient_path, f"{patient_dir}-t2f.nii.gz")
            t2w_file = os.path.join(patient_path, f"{patient_dir}-t2w.nii.gz")
            seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")

            # Check if all required files exist
            if all(os.path.exists(f) for f in [t1c_file, t1n_file, t2f_file, t2w_file, seg_file]):
                # Process and save the image files into the 'train_images' folder
                for image_file, image_type in zip([t1c_file, t1n_file, t2f_file, t2w_file],
                                                  ['t1c', 't1n', 't2f', 't2w']):
                    img = nib.load(image_file)
                    nib.save(img, os.path.join(train_images_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(train_masks_dir, f"{patient_dir}_seg.nii"))

    print("Train Files have been organized and extracted into images and masks directories.")


def val_data():
    brats_data_dir_2024 = data_dir
    val_images_dir = os.path.join(parent_dir, 'validation_data', 'validation_images')
    val_mask_dir = os.path.join(parent_dir, 'validation_data', 'validation_masks')

    # Ensure the output directories exist
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    for patient_dir in validation_directories:
        patient_path = os.path.join(brats_data_dir_2024, patient_dir)

        # Check if it's a directory and follows the naming convention
        if os.path.isdir(patient_path) and patient_dir.startswith('BraTS-GLI'):
            # Define the files for this patient
            t1c_file = os.path.join(patient_path, f"{patient_dir}-t1c.nii.gz")
            t1n_file = os.path.join(patient_path, f"{patient_dir}-t1n.nii.gz")
            t2f_file = os.path.join(patient_path, f"{patient_dir}-t2f.nii.gz")
            t2w_file = os.path.join(patient_path, f"{patient_dir}-t2w.nii.gz")
            seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")

            # Check if all required files exist
            if all(os.path.exists(f) for f in [t1c_file, t1n_file, t2f_file, t2w_file, seg_file]):
                # Process and save the image files into the 'train_images' folder
                for image_file, image_type in zip([t1c_file, t1n_file, t2f_file, t2w_file],
                                                  ['t1c', 't1n', 't2f', 't2w']):
                    img = nib.load(image_file)
                    nib.save(img, os.path.join(val_images_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(val_mask_dir, f"{patient_dir}_seg.nii"))

    print("Validation Files have been organized and extracted into images and masks directories.")


brats_2024()
val_data()
