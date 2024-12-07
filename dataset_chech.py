import pandas as pd
import os

# Define paths for the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))
training_csv_path = os.path.join(script_dir, 'whole_data.csv')
train_csv_path = os.path.join(script_dir, 'train.csv')
test_csv_path = os.path.join(script_dir, 'test.csv')

# Load the CSV files
print("Loading CSV files...")
training_data = pd.read_csv(training_csv_path)
train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Get the lengths of the dataframes
total_training_length = len(training_data)
train_length = len(train_data)
test_length = len(test_data)

print(f"Total samples in 'training_data.csv': {total_training_length}")
print(f"Samples in 'train.csv': {train_length}")
print(f"Samples in 'test.csv': {test_length}")

# Check for common samples between train and test
train_subjects = set(train_data['Subject'])
test_subjects = set(test_data['Subject'])
common_subjects = train_subjects.intersection(test_subjects)

if len(common_subjects) > 0:
    print(f"Common samples found between 'train.csv' and 'test.csv': {len(common_subjects)}")
    print(f"Common subjects: {common_subjects}")
else:
    print("No common samples between 'train.csv' and 'test.csv'.")

# Check if combining train and test gives the complete dataset
combined_subjects = train_subjects.union(test_subjects)
if set(training_data['Subject']) == combined_subjects:
    print("Combining 'train.csv' and 'test.csv' results in the complete dataset.")
else:
    print("Combining 'train.csv' and 'test.csv' does NOT result in the complete dataset.")
    missing_subjects = set(training_data['Subject']) - combined_subjects
    print(f"Missing subjects: {missing_subjects}")
