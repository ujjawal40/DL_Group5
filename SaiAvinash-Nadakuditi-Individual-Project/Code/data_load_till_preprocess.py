import os
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the Code folder
dl_group5_dir = os.path.dirname(script_dir)  # Path to DL_Group5 folder
data_dir = os.path.join(dl_group5_dir, 'Data')  # Correct path to Data folder

# Paths for train and test CSV files
train_csv_path = os.path.join(script_dir, 'train.csv')
test_csv_path = os.path.join(script_dir, 'test.csv')

# Subdirectories for train/test images and masks
train_image_dir = os.path.join(data_dir, 'train_images')
train_mask_dir = os.path.join(data_dir, 'train_masks')
test_image_dir = os.path.join(data_dir, 'test_images')
test_mask_dir = os.path.join(data_dir, 'test_masks')

# Ensure output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

# Supported modalities
modalities = ['t1c', 't1n', 't2f', 't2w', 'seg']


def gather_subject_data(data_dir):
    """
    Gather file paths for each subject from the data directory.
    """
    subject_data = []
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if os.path.isdir(subject_path):
            subject_info = {'Subject': subject}
            for file in os.listdir(subject_path):
                for modality in modalities:
                    if f"-{modality}" in file:
                        subject_info[modality] = os.path.join(subject_path, file)
            subject_data.append(subject_info)
    return pd.DataFrame(subject_data)


def validate_files(data):
    """
    Validate that all files exist and are readable.
    """
    for _, row in data.iterrows():
        for modality in modalities:
            if modality in row and os.path.exists(row[modality]):
                try:
                    nib.load(row[modality])
                except Exception as e:
                    print(f"Error loading {row[modality]}: {e}")


def organize_and_convert_files(data_csv, image_dir, mask_dir):
    """
    Organize files into train/test image and mask directories and convert .nii.gz to .nii in-place.
    """
    data = pd.read_csv(data_csv)
    for _, row in data.iterrows():
        subject = row['Subject']

        # Move and convert image modalities to the image directory
        for modality in modalities[:-1]:  # Exclude 'seg' as it is a mask
            src_image_path = row[modality]
            dst_image_path = os.path.join(image_dir, f"{subject}_{modality}.nii")
            convert_and_move_file(src_image_path, dst_image_path)

        # Move and convert segmentation mask to the mask directory
        src_mask_path = row['seg']
        dst_mask_path = os.path.join(mask_dir, f"{subject}_seg.nii")
        convert_and_move_file(src_mask_path, dst_mask_path)


def convert_and_move_file(src_path, dst_path):
    """
    Convert a .nii.gz file to a .nii file and move it to the destination.
    """
    try:
        img = nib.load(src_path)  # Load the .nii.gz file
        nib.save(img, dst_path)  # Save it as a .nii file
        os.remove(src_path)  # Remove the original .nii.gz file
        print(f"Converted and moved: {src_path} -> {dst_path}")
    except Exception as e:
        print(f"Error processing {src_path}: {e}")


# Main pipeline
def main():
    # Step 1: Gather and validate data
    print("Gathering training data...")
    whole_data = gather_subject_data(data_dir)

    print("Validating training files...")
    validate_files(whole_data)

    # Step 2: Split data into train and test sets
    train_data, test_data = train_test_split(whole_data, test_size=0.2, random_state=4001)
    train_data.to_csv(train_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)
    print(f"Train and test CSV files created: {train_csv_path}, {test_csv_path}")

    # Step 3: Organize train and test files with conversion
    print("Organizing and converting train data...")
    organize_and_convert_files(train_csv_path, train_image_dir, train_mask_dir)
    print("Organizing and converting test data...")
    organize_and_convert_files(test_csv_path, test_image_dir, test_mask_dir)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
