from colorama import Fore, Back, Style
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import nibabel as nib

c_ = Fore.GREEN
sr_ = Style.RESET_ALL
import torch
from tqdm import tqdm
tqdm.pandas()
import copy
from collections import defaultdict
import gc

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# PyTorch
import torch.nn as nn
from torch.optim import lr_scheduler, optimizer, Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from glob import glob
import cv2
import pandas as pd
import numpy as np
import random
import torch.nn.functional as F
import os

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

seed = 42
debug = False
model_name = 'Unet'
train_bs = 128
valid_bs = train_bs * 2
img_size = (224, 224)
n_epochs = 10
LR = 0.0001
scheduler = 'CosineAnnealingLR'
n_accumulate = max(1, 32 // train_bs)
n_fold = 5
fold_selected = 1
num_classes = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DIR = os.getcwd() + '/data/'


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')


# Directories
data_dir = '/home/ubuntu/DL_Group5/training_data1_v2'

# Read and shuffle subdirectories
sub_directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
random.shuffle(sub_directories)

# Split into train (80%) and validation (20%)
num_train = int(0.8 * len(sub_directories))
train_directories = sub_directories[:num_train]
validation_directories = sub_directories[num_train:]

print(f"Total directories: {len(sub_directories)}")
print(f"Training directories: {len(train_directories)}")
print(f"Validation directories: {len(validation_directories)}")


# Dataset
class BuildDataset(Dataset):
    def __init__(self, directories, data_dir, subset="train", transforms=None):
        self.directories = directories
        self.data_dir = data_dir
        self.subset = subset
        self.transforms = transforms

    def __len__(self):
        return len(self.directories)

    def __getitem__(self, index):
        # Get the directory for this sample
        patient_dir = self.directories[index]
        patient_path = os.path.join(self.data_dir, patient_dir)

        # Define file paths
        t1c_file = os.path.join(patient_path, f"{patient_dir}-t1c.nii.gz")
        seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")

        # Load the image and mask
        img = nib.load(t1c_file).get_fdata()
        mask = nib.load(seg_file).get_fdata()

        # Handle 3D masks by taking the middle slice or performing a projection
        if mask.ndim == 3:  # Check if mask is 3D
            mask = mask[:, :, mask.shape[2] // 2]  # Select the middle slice
            # Alternatively, perform a projection:
            # mask = mask.max(axis=2)

        # Resize to match model input size
        img = cv2.resize(img[:, :, 0], (128, 128))  # Take the first slice from the 3rd dimension
        mask = cv2.resize(mask, (128, 128))

        # Normalize the image
        img = np.expand_dims(img, axis=-1).astype(np.float32) / 255.0

        img = np.repeat(img, 3, axis=-1)

        # print(f"Image shape before permute: {img.shape}")
        # print(f"Mask shape before permute: {mask.shape}")

        # Convert mask to one-hot encoding (3 classes)
        masks = np.zeros((128, 128, 3))
        for i in range(3):
            masks[:, :, i] = (mask == i).astype(np.float32)

        # Apply transformations (if any)
        if self.transforms:
            augmented = self.transforms(image=img, mask=masks)
            img = augmented["image"]
            masks = augmented["mask"]

        return torch.tensor(img).permute(2, 0, 1), torch.tensor(masks).permute(2, 0, 1)


def read_data():
    """
    Reads the dataset and creates DataLoaders for training and validation sets.
    """
    train_dataset = BuildDataset(train_directories, data_dir, subset="train")
    val_dataset = BuildDataset(validation_directories, data_dir, subset="val")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    return train_loader, val_loader


def __load_gray_img(self, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0
    return img


def __load_img(self, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = cv2.resize(img, img_size)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype(np.float32) / 255.0
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')


def plot_batch(imgs, msks=None, size=5):
    """
    Plot a batch of images and masks side by side for visualization.
    """
    # Ensure size does not exceed batch size
    size = min(size, len(imgs))

    # Set up the figure
    plt.figure(figsize=(size * 5, 10))
    for idx in range(size):
        # Plot the image
        plt.subplot(2, size, idx + 1)
        img = imgs[idx].permute((1, 2, 0)).cpu().numpy()  # Convert to HWC format
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] range
        img = img.squeeze()  # Remove the channel dimension for grayscale
        plt.imshow(img, cmap="gray")  # Use gray colormap for better visualization
        plt.axis("off")
        plt.title("Image")

        # Plot the mask if available
        if msks is not None:
            plt.subplot(2, size, idx + 1 + size)
            mask = msks[idx].permute((1, 2, 0)).cpu().numpy()  # Convert to HWC format
            mask_combined = mask.sum(axis=-1)  # Combine classes for better visualization
            plt.imshow(mask_combined, cmap="viridis", alpha=0.8)
            plt.axis("off")
            plt.title("Mask")

    plt.tight_layout()
    plt.show()


# Model Architecture
class FCNVGG(nn.Module):
    def __init__(self):
        super(FCNVGG, self).__init__()

        vgg16 = models.vgg16(pretrained=True)
        self.vgg_features = vgg16.features

        # Freeze the VGG16 layers
        for param in self.vgg_features.parameters():
            param.requires_grad = False

        # Additional layers
        input_size = 7
        target_size = 224

        # Calculate the kernel_size and padding
        kernel_size = 2 * (target_size - input_size) + 1
        padding = kernel_size // 2

        # Define conv5 with adjusted parameters
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=3, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.conv5.weight)

        # Transpose convolution layers
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv1.weight)

        self.trans_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv2.weight)

        self.trans_conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv3.weight)

        self.trans_conv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.trans_conv4.weight)

        self.conv6 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)

        self.conv7 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv7.weight)

    def forward(self, x):
        x = self.vgg_features(x)
        x = self.conv5(x)
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        x = self.trans_conv3(x)
        x = self.trans_conv4(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class FCN8s(nn.Module):

    def __init__(self, n_class=3):
        super(FCN8s, self).__init__()
        self.features_123 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )
        self.features_4 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )
        self.features_5 = nn.Sequential(
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # fc7
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            # score_fr
            nn.Conv2d(4096, n_class, 1),
        )
        self.score_feat3 = nn.Conv2d(256, n_class, 1)
        self.score_feat4 = nn.Conv2d(512, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                              bias=False)
        self.upscore_4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)
        self.upscore_5 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False)

    def forward(self, x):
        feat3 = self.features_123(x)  #1/8
        feat4 = self.features_4(feat3)  #1/16
        feat5 = self.features_5(feat4)  #1/32

        score5 = self.classifier(feat5)
        upscore5 = self.upscore_5(score5)
        score4 = self.score_feat4(feat4)
        score4 = score4[:, :, 5:5+upscore5.size()[2], 5:5+upscore5.size()[3]].contiguous()
        score4 += upscore5

        score3 = self.score_feat3(feat3)
        upscore4 = self.upscore_4(score4)
        score3 = score3[:, :, 9:9+upscore4.size()[2], 9:9+upscore4.size()[3]].contiguous()
        score3 += upscore4
        h = self.upscore(score3)
        h = h[:, :, 28:28+x.size()[2], 28:28+x.size()[3]].contiguous()

        return h
def model_definition():
    model = FCNVGG()
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
    save_model(model)

    return model, optimizer, criterion, scheduler

def load_model(path):
    model, optimizer, criterion, scheduler = model_definition()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Calculating Loss
def dice_loss(predicted, target, epsilon=1e-6):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice_coefficient = (2.0 * intersection + epsilon) / (union + epsilon)
    return 1.0 - dice_coefficient

def iou_losOs(predicted, target, epsilon=1e-6):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return 1.0 - iou


def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return 0.5*iou_coef(y_pred, y_true) + 0.5*dice_coef(y_pred, y_true)
    #return dice_loss(y_pred, y_true)

class AdagradOptimizer(Optimizer):
    def __init__(self, params, lr=LR, eps=1e-8):
        defaults = dict(lr=lr, eps=eps)
        super(AdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                # Update parameters
                sum_ = state['sum']
                sum_.add_(grad ** 2)
                std = sum_.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], grad, std)

        return loss

# Train function
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    with tqdm(enumerate(dataloader), total=len(train_loader), desc="Epoch {}".format(epoch)) as pbar:
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            with amp.autocast(enabled=True):
                y_pred = model(images)  # Model's output
                # Resize the target to match y_pred's dimensions
                target_masks_resized = F.interpolate(masks, size=y_pred.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(y_pred, target_masks_resized)
                loss = loss / n_accumulate

            scaler.scale(loss).backward()

            if (step + 1) % n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()

                # zero the parameter gradients
                optimizer.zero_grad()

                # if scheduler is not None:
                #     val_loss, _ = valid_one_epoch(model, valid_loader, device=device, epoch=epoch)
                #     scheduler.step(val_loss)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                             lr=f'{current_lr:0.5f}',
                             gpu_mem=f'{mem:0.2f} GB')

            pbar.update(1)
            pbar.set_postfix_str("Train Loss: {:.5f}".format(epoch_loss))
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs, train_loader, val_loader):
    # To automatically log gradients

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        model.train()
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')

        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=device, epoch=epoch)

        val_loss, val_scores = valid_one_epoch(model, val_loader,
                                               device=device,
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores

        # Log the metrics

        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            # run.summary["Best Dice"]    = best_dice
            # run.summary["Best Jaccard"] = best_jaccard
            # run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model_Kanishk.pt")
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        last_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "final_model_Kanishk.pt")
        print()

    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


# Validation
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch=1):
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    dataset_size = 0
    running_loss = 0.0

    val_scores = []
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}") as pbar:
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            y_pred = model(images)  # Forward pass
            # Resize masks to match y_pred dimensions
            target_masks_resized = F.interpolate(masks, size=y_pred.shape[2:], mode='bilinear', align_corners=False)
            loss = criterion(y_pred, target_masks_resized)  # Compute loss

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            # Calculate metrics
            y_pred = nn.Sigmoid()(y_pred)  # Apply sigmoid for probabilities
            val_dice = dice_coef(target_masks_resized, y_pred).cpu().detach().numpy()
            val_jaccard = iou_coef(target_masks_resized, y_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{epoch_loss:.4f}', lr=f'{current_lr:.5f}', gpu_mem=f'{mem:.2f} GB')

    val_scores = np.mean(val_scores, axis=0)  # Average metrics across batches
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


def save_model(model):
    # Open the file

    print(model, file=open('summary_{}.txt'.format('kanishk'), "w"))


if __name__ == '__main__':
    # Set seed for reproducibility
    set_seed(seed)

    # Define the model, optimizer, criterion, and scheduler
    model, optimizer, criterion, scheduler = model_definition()

    # Read the data (train and validation loaders)
    train_loader, val_loader = read_data()

    # Train the model
    model, history = run_training(model, optimizer, scheduler, device, n_epochs, train_loader, val_loader)

    # Visualize a batch of training data
    for imgs, msks in train_loader:
        plot_batch(imgs, msks, size=5)
        break
