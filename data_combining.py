import nibabel as nib
import numpy as np
import os
import re


dir1='/home/ubuntu/DL_Group5/training_data1_v2'

dir2='/home/ubuntu/Code/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'  #Brats 2023

five_digit_numbers = []

# Filter filenames based on specific numeric values
filtered_filenames = []


images=os.listdir(dir2)
# print(images)


for name in images:
    match = re.search(r'BraTS-GLI-(\d{5})-\d{3}', name)
    if match:
        five_digit_number = match.group(1)  # Extract the five-digit number
        five_digit_numbers.append(five_digit_number)  # Append the number to the list

# print(five_digit_numbers)
integer_numbers = [int(num) for num in five_digit_numbers]
# print(integer_numbers)


# image_names_val=[]
# c=str(00000)
# for subject in range(0,700):
#     c = str(subject).zfill(5)
#     name = f'BraTS-GLI-{c}-000'
#     if(name in images):
#         image_names_val.append(name)
#
# print(len(image_names_val))
# print(image_names_val)
#
# image_names_test = []  # This will store valid image names that are found in 'images'
# for subject in range(701, 1301):  # Adjusted range from 500 to 800
#     c = str(subject).zfill(5)  # Convert subject number to a 5-digit string
#     name = f'BraTS-GLI-{c}-000'  # Construct the filename
#     if name in images:
#         image_names_test.append(name)  # Append the name if it exists in the images list
#
#
# print(len(image_names_test))
# print(image_names_test)


# image_names_train = []  # This will store valid image names that are found in 'images'
# for subject in range(1300,1700):  # Adjusted range from 500 to 800
#     c = str(subject).zfill(5)  # Convert subject number to a 5-digit string
#     name = f'BraTS-GLI-{c}-000'  # Construct the filename
#     if name in images:
#         image_names_train.append(name)  # Append the name if it exists in the images list
#
# print(len(image_names_train))
# print(image_names_train)


numbers=sorted(integer_numbers)
first_part = numbers[:500]
second_part = numbers[500:800]
third_part = numbers[800:]

def splitting(part):
    image_names=[]
    for subject in part:
        c = str(subject).zfill(5)
        name = f'BraTS-GLI-{c}-000'
        if (name in images):
            image_names.append(name)


    return image_names

validation_names=splitting(first_part)
test_names=splitting(second_part)
train_names=splitting(third_part)
print(len(train_names))
print(len(test_names))
print(len(validation_names))



#---------------------------------------------------------------------------------
#


def brats_2023():
    brats_data_dir_2024 = '/home/ubuntu/DL_Group5/training_data1_v2'
    brats_data_dir_2023 = '/home/ubuntu/DL_Group5/Code/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    train_dir = '/home/ubuntu/Dataset/combined_data/train_images'
    mask_dir = '/home/ubuntu/Dataset/combined_data/masks'



    # Ensure the output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)


    for patient_dir in train_names:
        patient_path = os.path.join(brats_data_dir_2023, patient_dir)

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
                for image_file, image_type in zip([t1c_file, t1n_file, t2f_file, t2w_file], ['t1c', 't1n', 't2f', 't2w']):
                    img = nib.load(image_file)
                    nib.save(img, os.path.join(train_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(mask_dir, f"{patient_dir}_seg.nii"))

    print("Files have been organized and extracted into images and masks directories.")

def brats_2024():
    brats_data_dir_2024 = '/home/ubuntu/DL_Group5/training_data1_v2'

    train_dir = '/home/ubuntu/Dataset/combined_data/train_images'
    mask_dir = '/home/ubuntu/Dataset/combined_data/masks'

    # Ensure the output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for patient_dir in os.listdir(brats_data_dir_2024):
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
                    nib.save(img, os.path.join(train_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(mask_dir, f"{patient_dir}_seg.nii"))

    print("Files have been organized and extracted into images and masks directories.")

def val_data():
    brats_data_dir_2023 = '/home/ubuntu/Dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    val_dir = '/home/ubuntu/Dataset/val_data/val_images'
    mask_dir = '/home/ubuntu/Dataset/val_data/val_masks'

    # Ensure the output directories exist
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for patient_dir in validation_names:
        patient_path = os.path.join(brats_data_dir_2023, patient_dir)

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
                    nib.save(img, os.path.join(val_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(mask_dir, f"{patient_dir}_seg.nii"))

    print("Validation Files have been organized and extracted into images and masks directories.")

def test_data():
    brats_data_dir_2023 = '/home/ubuntu/Code/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    test_dir = '/home/ubuntu/Dataset/test_data/test_images'
    mask_dir = '/home/ubuntu/Dataset/test_data/test_masks'

    # Ensure the output directories exist
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for patient_dir in test_names:
        patient_path = os.path.join(brats_data_dir_2023, patient_dir)

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
                    nib.save(img, os.path.join(test_dir, f"{patient_dir}_{image_type}.nii"))

                # Process and save the mask file into the 'masks' folder
                mask_img = nib.load(seg_file)
            nib.save(mask_img, os.path.join(mask_dir, f"{patient_dir}_seg.nii"))

    print("Test Files have been organized and extracted into images and masks directories.")


val_data()
test_data()





