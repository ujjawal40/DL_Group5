import os
import shutil


# Path to the main dataset directory
dataset_dir = '/home/ubuntu/DL-PROJECT/training_data1_v2/'

# Paths for the new image and mask directories
images_dir = os.path.join(dataset_dir, 'data/images')
masks_dir = os.path.join(dataset_dir, 'data/masks')

# Loop through all the patient directories in the main dataset directory
for patient_dir in os.listdir(dataset_dir):
    patient_path = os.path.join(dataset_dir, patient_dir)

    # Check if it's a directory
    if os.path.isdir(patient_path) and patient_dir.startswith('BraTS-GLI'):
        # Define the files for this patient
        t1c_file = os.path.join(patient_path, f"{patient_dir}-t1c.nii.gz")
        t1n_file = os.path.join(patient_path, f"{patient_dir}-t1n.nii.gz")
        t2f_file = os.path.join(patient_path, f"{patient_dir}-t2f.nii.gz")
        t2w_file = os.path.join(patient_path, f"{patient_dir}-t2w.nii.gz")
        seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")

        # Check if all required files exist
        if os.path.exists(t1c_file) and os.path.exists(t1n_file) and os.path.exists(t2f_file) and os.path.exists(
                t2w_file) and os.path.exists(seg_file):
            # Move the image files into the 'images' folder
            shutil.copy(t1c_file, os.path.join(images_dir, f"{patient_dir}_t1c.nii.gz"))
            shutil.copy(t1n_file, os.path.join(images_dir, f"{patient_dir}_t1n.nii.gz"))
            shutil.copy(t2f_file, os.path.join(images_dir, f"{patient_dir}_t2f.nii.gz"))
            shutil.copy(t2w_file, os.path.join(images_dir, f"{patient_dir}_t2w.nii.gz"))

            # Move the mask file into the 'masks' folder
            shutil.copy(seg_file, os.path.join(masks_dir, f"{patient_dir}_seg.nii.gz"))

print("Files have been organized into images and masks directories.")
