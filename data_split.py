import os
import pandas as pd
import shutil

# Define paths to the Data folder and the training_data1_v2 directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the Code folder
base_data_dir = os.path.join(script_dir, '../Data')  # Path to the Data folder
data_dir = os.path.join(base_data_dir, 'training_data1_v2')  # Path to training_data1_v2

# Paths for train and test CSV files
train_csv_path = os.path.join(script_dir, 'train.csv')  # CSV files are in the Code folder
test_csv_path = os.path.join(script_dir, 'test.csv')

# Define output directories within training_data1_v2
train_image_dir = os.path.join(data_dir, 'train_images')
train_mask_dir = os.path.join(data_dir, 'train_masks')
test_image_dir = os.path.join(data_dir, 'test_images')
test_mask_dir = os.path.join(data_dir, 'test_masks')

# Ensure output directories exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

# Supported modalities (excluding 'seg', which is treated as a mask)
modalities = ['t1c', 't1n', 't2f', 't2w']


def organize_files(data_csv, image_dir, mask_dir):
    """
    Organize files into separate image and mask directories within training_data1_v2.

    Args:
        data_csv (str): Path to the CSV file (train or test).
        image_dir (str): Directory to save image files.
        mask_dir (str): Directory to save mask files.
    """
    # Load the CSV file
    data = pd.read_csv(data_csv)

    # Iterate through each subject in the CSV
    for _, row in data.iterrows():
        subject = row['Subject']

        # Copy image modalities to the image directory
        for modality in modalities:
            src_image_path = row[modality]
            dst_image_path = os.path.join(image_dir, f"{subject}_{modality}.nii")
            shutil.move(src_image_path, dst_image_path)  # Move file to destination

        # Copy segmentation mask to the mask directory
        src_mask_path = row['seg']
        dst_mask_path = os.path.join(mask_dir, f"{subject}_seg.nii")
        shutil.move(src_mask_path, dst_mask_path)  # Move file to destination


# Organize train files
print("Organizing train data...")
organize_files(train_csv_path, train_image_dir, train_mask_dir)

# Organize test files
print("Organizing test data...")
organize_files(test_csv_path, test_image_dir, test_mask_dir)

print("Data organization completed!. Proceed Further!.")
