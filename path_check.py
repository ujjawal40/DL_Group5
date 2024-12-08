import SimpleITK as sitk
try:
    img = sitk.ReadImage("/home/ubuntu/DL_Group5/Data/training_data1_v2/train_images/BraTS-GLI-03064-100_t2w.nii")

    print("File loaded successfully.")
    print(img)
except Exception as e:
    print(f"Failed to load file: {e}")


# import nibabel as nib
#
# input_path = "/home/ubuntu/DL_Group5/Data/training_data1_v2/train_images/BraTS-GLI-00517-100_t1c.nii"
# output_path = "/home/ubuntu/DL_Group5/Data/training_data1_v2/train_images/cleaned_BraTS-GLI-00517-100_t1c.nii"
#
# try:
#     img = nib.load(input_path)
#     print("File loaded successfully. Nibs")
#     # nib.save(img, output_path)
#     # print(f"File re-saved to {output_path}")
# except Exception as e:
#     print(f"Error: {e}")

