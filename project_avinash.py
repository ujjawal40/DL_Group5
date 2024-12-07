#%% --------------------------------------- Imports --------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib  # For reading .nii files
import numpy as np
import torch.nn.functional as F

#%% --------------------------------------- Data Prep ------------------------------------------------------------------
class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._get_samples()

    def _get_samples(self):
        samples = []
        for patient_folder in os.listdir(self.root_dir):
            modalities = []
            folder_path = os.path.join(self.root_dir, patient_folder)
            if os.path.isdir(folder_path):
                for filename in sorted(os.listdir(folder_path)):
                    if filename.endswith('.nii'):
                        file_path = os.path.join(folder_path, filename)
                        modalities.append(file_path)
                if len(modalities) == 5:
                    samples.append((modalities[:4], modalities[4]))  # (4 modalities, 1 segmentation mask)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        modality_paths, mask_path = self.samples[idx]

        # Load modalities as a 4D tensor (4, H, W, D)
        modalities = [nib.load(mod).get_fdata() for mod in modality_paths]
        x = np.stack(modalities)  # Shape: (4, 240, 240, 155)

        # Load mask and reshape to (1, H, W, D) for segmentation output
        y = nib.load(mask_path).get_fdata()
        y = y[np.newaxis, :, :, :]  # Shape: (1, 240, 240, 155)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)  # Assuming mask labels are integers

        if self.transform:
            x, y = self.transform(x, y)

        return x, y

# Paths to training and validation data
train_data = BrainTumorDataset(root_dir='../Data/training_data1_v2')
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)


#%% --------------------------------------- Model Definiton ------------------------------------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, out_channels, kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Instantiate the model
model = UNet3D(in_channels=4, out_channels=2)  # Assuming 2 classes: tumor vs. non-tumor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
model = model.to(device)


#%% --------------------------------------- Model Training -------------------------------------------------------------
# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()  # Use cross-entropy for segmentation tasks
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(train_loader):

        inputs, masks = inputs.to(device), masks.to(device)

        # Forward pass
        outputs = model(inputs)
        # Assuming `outputs` is the model output and `masks` is the ground truth mask
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        masks = masks.squeeze(1)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


#%% --------------------------------------- Validation Loop ------------------------------------------------------------
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(val_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            # Reshape masks for compatibility
            masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
            masks = masks.squeeze(1)

            # Compute validation loss
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Paths to validation data
val_data = BrainTumorDataset(root_dir='../Data/validation_data1_v2')
val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

#%% --------------------------------------- Training and Validation Integration ----------------------------------------
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(train_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)

        # Reshape masks
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        masks = masks.squeeze(1)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loss
    val_loss = validate_model(model, val_loader, criterion)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

#%% --------------------------------------- Model Testing --------------------------------------------------------------
def test_model(model, test_loader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
    return all_outputs

# Paths to testing data
test_data = BrainTumorDataset(root_dir='../Data/test_data1_v2')
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Load best model
model.load_state_dict(torch.load("best_model.pth"))

# Perform testing
test_outputs = test_model(model, test_loader)
print(f"Test completed on {len(test_outputs)} samples.")

#%% --------------------------------------- Dice Coefficient -----------------------------------------------------------
def dice_coefficient(preds, targets, smooth=1e-6):
    preds = preds.argmax(dim=1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

# Evaluate Dice score on validation set
dice_scores = []
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        dice = dice_coefficient(outputs, masks)
        dice_scores.append(dice)

avg_dice_score = sum(dice_scores) / len(dice_scores)
print(f"Average Dice Coefficient on Validation Set: {avg_dice_score}")

#%% --------------------------------------- Visualization --------------------------------------------------------------
import matplotlib.pyplot as plt

def visualize_sample(input_tensor, mask_tensor, pred_tensor):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display one slice from each tensor
    slice_idx = input_tensor.shape[-1] // 2

    ax[0].imshow(input_tensor[0, :, :, slice_idx], cmap='gray')
    ax[0].set_title('Input Slice')

    ax[1].imshow(mask_tensor[0, :, :, slice_idx], cmap='jet')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(pred_tensor[0, :, :, slice_idx], cmap='jet')
    ax[2].set_title('Prediction')

    plt.show()

# Visualize a random sample from validation set
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1, keepdim=True)
        visualize_sample(inputs.cpu().numpy()[0], masks.cpu().numpy(), outputs.cpu().numpy()[0])
        break


print("For checking purposes")
print("For checking purposes")
print("Avinash")
print("Avinash")
print("Avinash")
print("This is a change done online")


#%% --------------------------------------- Data Augmentation ----------------------------------------------------------
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

class Augmentations:
    def __call__(self, x, y):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1)  # Horizontal flip
            y = np.flip(y, axis=1)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2)  # Vertical flip
            y = np.flip(y, axis=2)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)  # Random rotation
            x = np.rot90(x, axes=(1, 2))
            y = np.rot90(y, axes=(1, 2))
        return x, y

# Applying augmentations
augmented_train_data = BrainTumorDataset(root_dir='../Data/training_data1_v2', transform=Augmentations())
augmented_train_loader = DataLoader(augmented_train_data, batch_size=1, shuffle=True)

#%% --------------------------------------- Learning Rate Scheduler ----------------------------------------------------
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

#%% --------------------------------------- Enhanced Training Loop -----------------------------------------------------
best_dice_score = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks) in enumerate(augmented_train_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)

        # Reshape masks
        masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
        masks = masks.squeeze(1)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Step the learning rate scheduler
    scheduler.step()

    # Validation loss and Dice score
    val_loss = validate_model(model, val_loader, criterion)
    val_dice_scores = []
    model.eval()
    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            masks = F.interpolate(masks.float(), size=(180, 216, 180), mode="nearest").long()
            val_dice = dice_coefficient(outputs, masks)
            val_dice_scores.append(val_dice)
    avg_val_dice = sum(val_dice_scores) / len(val_dice_scores)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {running_loss / len(augmented_train_loader)}, "
          f"Validation Loss: {val_loss}, Avg Dice Score: {avg_val_dice}")

    # Save model with best Dice score
    if avg_val_dice > best_dice_score:
        best_dice_score = avg_val_dice
        torch.save(model.state_dict(), "best_dice_model.pth")
        print("Best Dice model saved!")

#%% --------------------------------------- Metrics Tracking -----------------------------------------------------------
import json

metrics = {
    "Best Dice Score": best_dice_score,
    "Validation Loss": val_loss
}

with open("training_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Training metrics saved!")

#%% --------------------------------------- Post-Processing ------------------------------------------------------------
def post_process_predictions(pred_tensor, threshold=0.5):
    pred_tensor = pred_tensor.argmax(dim=1, keepdim=True)
    binary_preds = (pred_tensor > threshold).long()
    return binary_preds

# Post-process and save test predictions
post_processed_preds = []
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        post_processed = post_process_predictions(outputs)
        post_processed_preds.append(post_processed.cpu().numpy())

np.save("test_predictions.npy", post_processed_preds)
print("Post-processed predictions saved!")

#%% --------------------------------------- Visualization Improvements -------------------------------------------------
def visualize_multiple_slices(input_tensor, mask_tensor, pred_tensor, num_slices=3):
    fig, axs = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))

    slices = np.linspace(0, input_tensor.shape[-1] - 1, num_slices, dtype=int)

    for i, slice_idx in enumerate(slices):
        axs[i, 0].imshow(input_tensor[0, :, :, slice_idx], cmap='gray')
        axs[i, 0].set_title(f'Input Slice {slice_idx}')

        axs[i, 1].imshow(mask_tensor[0, :, :, slice_idx], cmap='jet')
        axs[i, 1].set_title(f'Ground Truth Slice {slice_idx}')

        axs[i, 2].imshow(pred_tensor[0, :, :, slice_idx], cmap='jet')
        axs[i, 2].set_title(f'Prediction Slice {slice_idx}')

    plt.tight_layout()
    plt.show()

# Enhanced visualization for test data
model.eval()
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        outputs = outputs.argmax(dim=1, keepdim=True)
        visualize_multiple_slices(inputs.cpu().numpy()[0], masks.cpu().numpy(), outputs.cpu().numpy()[0], num_slices=5)
        break


#%% --------------------------------------- Post-Training Analysis: Model Evaluation -----------------------------------
import seaborn as sns

def calculate_class_distribution(dataset_loader):
    class_counts = torch.zeros(2)  # Assuming 2 classes: 0 (background), 1 (tumor)
    for _, masks in dataset_loader:
        masks = masks.flatten()
        for cls in range(2):
            class_counts[cls] += torch.sum(masks == cls)
    return class_counts / torch.sum(class_counts)

# Calculate class distributions in training, validation, and testing sets
train_class_dist = calculate_class_distribution(train_loader)
val_class_dist = calculate_class_distribution(val_loader)
test_class_dist = calculate_class_distribution(test_loader)

print(f"Class distribution (Training): {train_class_dist}")
print(f"Class distribution (Validation): {val_class_dist}")
print(f"Class distribution (Testing): {test_class_dist}")

# Plot class distributions
def plot_class_distribution(distributions, labels, title):
    sns.barplot(x=labels, y=distributions)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Proportion")
    plt.show()

plot_class_distribution(train_class_dist.numpy(), ["Background", "Tumor"], "Training Set Class Distribution")
plot_class_distribution(val_class_dist.numpy(), ["Background", "Tumor"], "Validation Set Class Distribution")
plot_class_distribution(test_class_dist.numpy(), ["Background", "Tumor"], "Testing Set Class Distribution")

#%% --------------------------------------- Per-Class Dice Score -------------------------------------------------------
def calculate_per_class_dice(model, dataset_loader):
    model.eval()
    dice_scores = torch.zeros(2)  # Assuming 2 classes
    with torch.no_grad():
        for inputs, masks in dataset_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            outputs = outputs.argmax(dim=1)
            for cls in range(2):
                preds = (outputs == cls).float()
                targets = (masks == cls).float()
                intersection = (preds * targets).sum()
                union = preds.sum() + targets.sum()
                dice = (2.0 * intersection) / (union + 1e-6)
                dice_scores[cls] += dice
    return dice_scores / len(dataset_loader)

train_dice_scores = calculate_per_class_dice(model, train_loader)
val_dice_scores = calculate_per_class_dice(model, val_loader)
test_dice_scores = calculate_per_class_dice(model, test_loader)

print(f"Per-Class Dice Scores (Training): {train_dice_scores}")
print(f"Per-Class Dice Scores (Validation): {val_dice_scores}")
print(f"Per-Class Dice Scores (Testing): {test_dice_scores}")

#%% --------------------------------------- Confusion Matrix -----------------------------------------------------------
from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_confusion_matrix(model, dataset_loader):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, masks in dataset_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).flatten().cpu().numpy()
            targets = masks.flatten().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    return confusion_matrix(all_targets, all_preds)

conf_matrix = generate_confusion_matrix(model, val_loader)
conf_df = pd.DataFrame(conf_matrix, index=["Background", "Tumor"], columns=["Background", "Tumor"])
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#%% --------------------------------------- Precision, Recall, F1-Score ------------------------------------------------
from sklearn.metrics import classification_report

def generate_classification_report(model, dataset_loader):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, masks in dataset_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).flatten().cpu().numpy()
            targets = masks.flatten().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    return classification_report(all_targets, all_preds, target_names=["Background", "Tumor"])

val_class_report = generate_classification_report(model, val_loader)
print("Validation Classification Report:")
print(val_class_report)

#%% --------------------------------------- Feature Importance ---------------------------------------------------------
def visualize_feature_importance(modality_importance, modality_names):
    plt.bar(modality_names, modality_importance)
    plt.title("Feature Importance Across Modalities")
    plt.ylabel("Importance Score")
    plt.show()

# Example modality importance analysis (Dummy example)
modality_importance = [0.8, 0.9, 0.7, 0.6]
modality_names = ["T1", "T1CE", "T2", "FLAIR"]
visualize_feature_importance(modality_importance, modality_names)

#%% --------------------------------------- Segmentation Boundary Analysis ---------------------------------------------
def boundary_analysis(preds, targets):
    edges_pred = np.gradient(preds.numpy())
    edges_target = np.gradient(targets.numpy())
    boundary_diff = np.sum(np.abs(edges_pred - edges_target))
    return boundary_diff

boundary_differences = []
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs).argmax(dim=1).cpu()
        boundary_diff = boundary_analysis(outputs, masks.cpu())
        boundary_differences.append(boundary_diff)

avg_boundary_diff = sum(boundary_differences) / len(boundary_differences)
print(f"Average Boundary Difference: {avg_boundary_diff}")

#%% --------------------------------------- Save Analysis Results ------------------------------------------------------
analysis_results = {
    "Training Class Distribution": train_class_dist.numpy().tolist(),
    "Validation Class Distribution": val_class_dist.numpy().tolist(),
    "Testing Class Distribution": test_class_dist.numpy().tolist(),
    "Per-Class Dice Scores (Training)": train_dice_scores.tolist(),
    "Per-Class Dice Scores (Validation)": val_dice_scores.tolist(),
    "Per-Class Dice Scores (Testing)": test_dice_scores.tolist(),
    "Average Boundary Difference": avg_boundary_diff
}

with open("post_training_analysis.json", "w") as f:
    json.dump(analysis_results, f)

print("Post-training analysis results saved!")


#%% --------------------------------------- ROC and AUC Analysis ------------------------------------------------------
from sklearn.metrics import roc_curve, auc

def calculate_roc_auc(model, dataset_loader):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, masks in dataset_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = outputs.softmax(dim=1)[:, 1].flatten().cpu().numpy()
            targets = (masks.flatten().cpu().numpy() == 1).astype(int)
            all_preds.extend(preds)
            all_targets.extend(targets)

    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

fpr, tpr, roc_auc = calculate_roc_auc(model, val_loader)
print(f"Validation ROC AUC: {roc_auc}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

#%% --------------------------------------- Precision-Recall Curve ----------------------------------------------------
from sklearn.metrics import precision_recall_curve

def calculate_precision_recall(model, dataset_loader):
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for inputs, masks in dataset_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            preds = outputs.softmax(dim=1)[:, 1].flatten().cpu().numpy()
            targets = (masks.flatten().cpu().numpy() == 1).astype(int)
            all_preds.extend(preds)
            all_targets.extend(targets)

    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    return precision, recall

precision, recall = calculate_precision_recall(model, val_loader)

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

#%% --------------------------------------- Detailed Class-Wise Metrics -----------------------------------------------
from sklearn.metrics import jaccard_score

def calculate_class_metrics(model, dataset_loader):
    model.eval()
    jaccard_scores = []
    for inputs, masks in dataset_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()

        for cls in range(2):  # Assuming two classes
            cls_jaccard = jaccard_score(targets.flatten() == cls, preds.flatten() == cls)
            jaccard_scores.append(cls_jaccard)

    return jaccard_scores

class_metrics = calculate_class_metrics(model, val_loader)
print(f"Class-Wise Jaccard Scores: {class_metrics}")

#%% --------------------------------------- Misclassified Regions Analysis --------------------------------------------
def analyze_misclassified_regions(preds, targets):
    error_map = (preds != targets).float()
    error_percentage = torch.sum(error_map) / torch.numel(targets)
    return error_map, error_percentage.item()

misclassified_maps = []
error_percentages = []
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs).argmax(dim=1)
        error_map, error_percentage = analyze_misclassified_regions(outputs, masks)
        misclassified_maps.append(error_map.cpu().numpy())
        error_percentages.append(error_percentage)

avg_error_percentage = sum(error_percentages) / len(error_percentages)
print(f"Average Misclassification Percentage: {avg_error_percentage:.2f}%")

#%% --------------------------------------- Interactive Visualization -------------------------------------------------
from ipywidgets import interact

def interactive_visualization(input_tensor, mask_tensor, pred_tensor):
    @interact(slice_idx=(0, input_tensor.shape[-1] - 1))
    def plot_slice(slice_idx=0):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(input_tensor[0, :, :, slice_idx], cmap='gray')
        ax[0].set_title('Input Slice')
        ax[1].imshow(mask_tensor[0, :, :, slice_idx], cmap='jet')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(pred_tensor[0, :, :, slice_idx], cmap='jet')
        ax[2].set_title('Prediction')
        plt.show()

# Use interactive visualization on a sample
model.eval()
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs).argmax(dim=1, keepdim=True)
        interactive_visualization(inputs.cpu().numpy()[0], masks.cpu().numpy(), outputs.cpu().numpy()[0])
        break

#%% --------------------------------------- Save Visualizations -------------------------------------------------------
import os

visualization_dir = "visualizations"
os.makedirs(visualization_dir, exist_ok=True)

def save_visualizations(input_tensor, mask_tensor, pred_tensor, sample_idx):
    for i in range(input_tensor.shape[-1]):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(input_tensor[0, :, :, i], cmap='gray')
        ax[0].set_title('Input Slice')
        ax[1].imshow(mask_tensor[0, :, :, i], cmap='jet')
        ax[1].set_title('Ground Truth')
        ax[2].imshow(pred_tensor[0, :, :, i], cmap='jet')
        ax[2].set_title('Prediction')
        plt.savefig(os.path.join(visualization_dir, f"sample_{sample_idx}_slice_{i}.png"))
        plt.close()

# Save visualizations for test samples
model.eval()
with torch.no_grad():
    for idx, (inputs, masks) in enumerate(test_loader):
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs).argmax(dim=1, keepdim=True)
        save_visualizations(inputs.cpu().numpy()[0], masks.cpu().numpy(), outputs.cpu().numpy()[0], idx)
        if idx == 4:  # Save visualizations for the first 5 samples
            break



#%% --------------------------------------- Boundary Distance Analysis ------------------------------------------------
from scipy.ndimage import distance_transform_edt

def boundary_distance(preds, targets):
    preds_boundary = distance_transform_edt(1 - preds) == 1
    targets_boundary = distance_transform_edt(1 - targets) == 1
    distance = np.abs(preds_boundary - targets_boundary)
    return distance.sum()

boundary_distances = []
model.eval()
with torch.no_grad():
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        outputs = model(inputs).argmax(dim=1).cpu().numpy()
        masks = masks.cpu().numpy()
        dist = boundary_distance(outputs[0], masks[0])
        boundary_distances.append(dist)

avg_boundary_distance = sum(boundary_distances) / len(boundary_distances)
print(f"Average Boundary Distance: {avg_boundary_distance:.2f}")

#%% --------------------------------------- Overlap and Dice Visualization --------------------------------------------
def plot_dice_vs_overlap(dice_scores, overlap_scores, title="Dice vs Overlap"):
    plt.scatter(overlap_scores, dice_scores, alpha=0.6)
    plt.xlabel("Overlap Score")
    plt.ylabel("Dice Score")
    plt.title(title)
    plt.show()

# Dummy data for overlap scores (generate real values from your model analysis)
overlap_scores = np.random.uniform(0.6, 0.9, len(dice_scores))

# Plot Dice vs Overlap
plot_dice_vs_overlap(dice_scores=val_dice_scores.tolist(), overlap_scores=overlap_scores)

#%% --------------------------------------- Metrics per Patient --------------------------------------------------------
def patient_wise_metrics(model, dataset_loader):
    patient_metrics = []
    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks) in enumerate(dataset_loader):
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs).argmax(dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            dice = dice_coefficient(torch.tensor(outputs), torch.tensor(masks))
            jaccard = jaccard_score(masks.flatten(), outputs.flatten())
            patient_metrics.append({"Patient": idx, "Dice": dice, "Jaccard": jaccard})
    return patient_metrics

metrics_per_patient = patient_wise_metrics(model, val_loader)
print(metrics_per_patient[:5])  # Display metrics for the first 5 patients

#%% --------------------------------------- Heatmap of Misclassified Regions -------------------------------------------
def plot_error_heatmap(error_map, title="Error Heatmap"):
    plt.imshow(error_map[0, :, :, error_map.shape[3] // 2], cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Example heatmap for one patient
example_error_map = misclassified_maps[0]
plot_error_heatmap(example_error_map, title="Example Misclassification Heatmap")

#%% --------------------------------------- 3D Visualization of Predictions --------------------------------------------
import plotly.graph_objects as go

def visualize_3d_segmentation(input_volume, pred_volume, mask_volume):
    fig = go.Figure()

    # Input Volume (Overlay as a grayscale background)
    fig.add_trace(go.Volume(
        x=np.arange(input_volume.shape[1]),
        y=np.arange(input_volume.shape[2]),
        z=np.arange(input_volume.shape[3]),
        value=input_volume[0],
        isomin=0.1,
        isomax=0.9,
        opacity=0.1,  # Adjust for transparency
        surface_count=15
    ))

    # Prediction Volume
    fig.add_trace(go.Volume(
        x=np.arange(pred_volume.shape[1]),
        y=np.arange(pred_volume.shape[2]),
        z=np.arange(pred_volume.shape[3]),
        value=pred_volume[0],
        isomin=0.5,
        isomax=1,
        opacity=0.3,
        surface_count=10,
        colorscale="Viridis"
    ))

    # Mask Volume
    fig.add_trace(go.Volume(
        x=np.arange(mask_volume.shape[1]),
        y=np.arange(mask_volume.shape[2]),
        z=np.arange(mask_volume.shape[3]),
        value=mask_volume[0],
        isomin=0.5,
        isomax=1,
        opacity=0.3,
        surface_count=10,
        colorscale="Reds"
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis'),
        title="3D Visualization of Segmentation"
    )
    fig.show()

# Test 3D Visualization
model.eval()
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.cpu().numpy(), masks.cpu().numpy()
        outputs = model(inputs.to(device)).argmax(dim=1).cpu().numpy()
        visualize_3d_segmentation(inputs[0], outputs, masks[0])
        break

#%% --------------------------------------- Explainability: Grad-CAM --------------------------------------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def grad_cam_analysis(model, inputs):
    target_layers = [model.encoder[-1]]  # Example: last layer of the encoder
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=inputs, targets=None)  # No specific target class for segmentation
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam

# Apply Grad-CAM on a test sample
model.eval()
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        cam_result = grad_cam_analysis(model, inputs)
        plt.imshow(cam_result, cmap='jet')
        plt.title("Grad-CAM Heatmap")
        plt.colorbar()
        plt.show()
        break

#%% --------------------------------------- Save Metrics and Visualizations -------------------------------------------
# Save patient-wise metrics
import pandas as pd
metrics_df = pd.DataFrame(metrics_per_patient)
metrics_df.to_csv("patient_metrics.csv", index=False)
print("Patient-wise metrics saved to patient_metrics.csv!")

# Save visualizations as HTML files
visualization_dir = "3d_visualizations"
os.makedirs(visualization_dir, exist_ok=True)

def save_3d_visualization(fig, filename):
    fig.write_html(os.path.join(visualization_dir, filename))

# Save example 3D visualization
model.eval()
with torch.no_grad():
    for inputs, masks in test_loader:
        inputs, masks = inputs.cpu().numpy(), masks.cpu().numpy()
        outputs = model(inputs.to(device)).argmax(dim=1).cpu().numpy()
        fig = visualize_3d_segmentation(inputs[0], outputs, masks[0])
        save_3d_visualization(fig, "example_3d_visualization.html")
        break
