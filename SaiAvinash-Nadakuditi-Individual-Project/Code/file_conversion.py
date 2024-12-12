import os
import nibabel as nib

# Paths to directories
script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the Code folder
dl_group5_dir = os.path.dirname(script_dir)  # Path to DL_Group5 folder
base_data_dir = os.path.join(dl_group5_dir, 'nData')  # Path to training_data1_v2

# Subdirectories to process
subdirs = ['train_images', 'train_masks', 'test_images', 'test_masks']


def convert_nii_gz_to_nii_in_place(directory):
    """
    Convert all .nii.gz files in a directory to .nii files in-place.
    Args:
        directory (str): Path to the directory to process.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".nii.gz"):
            file_path = os.path.join(directory, file_name)
            new_file_path = file_path.replace(".nii.gz", ".nii")

            try:
                # Load the .nii.gz file
                img = nib.load(file_path)

                # Save it as .nii file
                nib.save(img, new_file_path)

                # Remove the original .nii.gz file
                os.remove(file_path)
                print(f"Converted: {file_path} -> {new_file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


# Process each subdirectory
for subdir in subdirs:
    dir_path = os.path.join(base_data_dir, subdir)
    if os.path.isdir(dir_path):
        print(f"Processing directory: {dir_path}")
        convert_nii_gz_to_nii_in_place(dir_path)

print("Conversion completed for all directories.")
