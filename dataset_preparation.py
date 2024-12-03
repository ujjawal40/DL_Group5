import os
import nibabel as nib
import pandas as pd

# Define the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..', 'Data')
training_dir = os.path.join(base_dir, 'training_data1_v2')
validation_dir = os.path.join(base_dir, 'validation_data')

# Supported modalities
modalities = ['t1c', 't1n', 't2f', 't2w', 'seg']

# Function to gather file paths for each subject
def gather_subject_data(data_dir):
    subject_data = []
    for subject in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject)
        if os.path.isdir(subject_path):
            # Initialize a dictionary for the subject
            subject_info = {'Subject': subject}
            for file in os.listdir(subject_path):
                # Identify modality from the filename
                for modality in modalities:
                    if f"-{modality}" in file:
                        subject_info[modality] = os.path.join(subject_path, file)
            subject_data.append(subject_info)
    return pd.DataFrame(subject_data)

# Gather training and validation data
print("Gathering training data...")
training_data = gather_subject_data(training_dir)
print("Gathering validation data...")
validation_data = gather_subject_data(validation_dir)

# Print a summary
print(f"Training data contains {len(training_data)} subjects.")
print(f"Validation data contains {len(validation_data)} subjects.")

# Save the mapping to CSV for reference
training_data.to_csv('training_data_mapping.csv', index=False)
validation_data.to_csv('validation_data_mapping.csv', index=False)

# Validate that files are readable
def validate_files(data):
    for _, row in data.iterrows():
        for modality in modalities:
            if modality in row and os.path.exists(row[modality]):
                try:
                    # Load the NIfTI file
                    nib.load(row[modality])
                except Exception as e:
                    print(f"Error loading {row[modality]}: {e}")

# Validate training and validation files
print("Validating training files...")
validate_files(training_data)
print("Validating validation files...")
validate_files(validation_data)

print("Data preparation completed!")


