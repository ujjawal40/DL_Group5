# Import Libraries
import os
import nibabel as nib
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..', 'Data')
training_dir = os.path.join(base_dir, 'training_data1_v2')

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

# Gather all training data
print("Gathering training data...")
whole_data = gather_subject_data(training_dir)

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

print("Validating training files...")
validate_files(whole_data)

# Save the complete training data mapping to a CSV
data_csv_path = os.path.join(script_dir, 'whole_data.csv')
whole_data.to_csv(data_csv_path, index=False)
print(f"Training data saved to {data_csv_path}")

# Split the data into train and test sets (80-20 split) with a fixed random state
train_data, test_data = train_test_split(whole_data, test_size=0.2, random_state=4001)

# Save train and test data to separate CSV files
train_csv_path = os.path.join(script_dir, 'train.csv')
test_csv_path = os.path.join(script_dir, 'test.csv')

train_data.to_csv(train_csv_path, index=False)
test_data.to_csv(test_csv_path, index=False)

print(f"Train data saved to {train_csv_path}")
print(f"Test data saved to {test_csv_path}")

print("Data splitting completed!. Proceed further")





