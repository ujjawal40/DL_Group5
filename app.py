import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import SimpleITK as sitk
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# Inject CSS for styling
st.markdown("""
    <style>
    /* General body styling */
    body {
        font-family: "Arial", sans-serif;
        background-color: #f8f9fa;
        color: #333333;
    }

    /* Make the app use the full screen */
    .main {
        padding: 0rem;
        width: 100%;
    }

    /* Center the content */
    .block-container {
        padding: 1rem 2rem;
        max-width: 1200px;
        margin: auto;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        background: #ffffff;
        border-radius: 8px;
    }

    /* Title styles */
    h1 {
        color: #007bff;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    /* Header styles */
    h2, h3 {
        color: #333333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Subheader styles */
    h4, h5 {
        color: #555555;
        margin-bottom: 0.5rem;
    }

    /* Tumor region radio buttons */
    .stRadio > label {
        font-size: 1rem;
        color: #007bff;
        margin-bottom: 0.5rem;
    }

    /* Radio button styling */
    input[type=radio]:checked + div {
        color: #0056b3 !important;
        font-weight: bold;
    }

    /* Button hover */
    button:hover {
        background-color: #0056b3 !important;
        color: #ffffff !important;
    }

    /* Link colors */
    a {
        color: #007bff;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Center and style the title
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #007bff;">Advanced Brain Tumor Segmentation: Preprocessing, Augmentation, and Deep Learning Analysis</h1>
        <h2 style="color: #007bff;">Group 5</h2>
    </div>
""", unsafe_allow_html=True)

st.subheader("Team Members:")
st.write("- Ujjawal")
st.write("- Bala Krishna")
st.write("- Bhagawath Sai")
st.write("- Sai Avinash")


st.header("Project Relevance")
st.write("""
Brain tumor segmentation is a crucial task in medical imaging that assists in the diagnosis and treatment of patients. 
Accurate segmentation helps radiologists and clinicians understand tumor regions, identify malignancies, and monitor treatment progress.
The project aims to implement deep learnoing models (3D FCNNs, Unet, Residual Unet) to segment various regions of brain tumors from multi-modal MRI scans.
""")

st.header("Dataset Description")
st.write("""
This project utilizes the BraTS dataset, which contains pre-processed multi-modal MRI scans for brain tumor segmentation.
The dataset includes the following modalities:
- T1-weighted (t1n)
- T1-weighted post-contrast (t1c)
- T2-weighted (t2w)
- Fluid-attenuated inversion recovery (t2f)
""")

st.header("Types of Scans and Tumor Regions")
st.write("""
The dataset includes annotations for various tumor regions:
- **Whole Tumor (WT)**: All visible tumor regions.
- **Necrotic and Non-enhancing Tumor Core (NETC)**: Central dead tissue and non-enhancing regions.
- **Enhancing Tumor (ET)**: Actively growing tumor cells.
- **Surrounding Non-tumor Fluid Heterogeneity (SNFH)**: Swelling or fluid accumulation around the tumor.
- **Resection Cavity**: Post-surgical void in the brain.
""")

# Tumor regions selection
tumor_regions = {
    "Whole Tumor (WT)": "All visible tumor regions.",
    "Necrotic and Non-enhancing Tumor Core (NETC)": "Central dead tissue and non-enhancing regions.",
    "Enhancing Tumor (ET)": "Actively growing tumor cells.",
    "Surrounding Non-tumor Fluid Heterogeneity (SNFH)": "Swelling or fluid accumulation around the tumor.",
    "Resection Cavity": "Post-surgical void in the brain."
}

selected_region = st.radio("Select Tumor Region:", list(tumor_regions.keys()))
st.write(f"**Description of Selected Region:** {tumor_regions[selected_region]}")

st.header("Model Test")

# Directories
val_image_dir = "/home/ubuntu/DL-PROJECT/val_data/val_images"
val_mask_dir = "/home/ubuntu/DL-PROJECT/val_data/val_masks"
model_path = "/home/ubuntu/avinash_folder/best_model.pth"

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the UNet3D Model
# Define the U-Net model components
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // 2, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, try using reflect padding as an alternative
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# UNet3D Model Definition
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = UNet3D(n_channels=4, n_classes=5, bilinear=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

# Preprocessing Function
def preprocess_image(file_path, target_size=(132, 132, 116)):
    image = sitk.ReadImage(file_path, sitk.sitkFloat32)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing([image.GetSpacing()[0] * (image.GetSize()[0] / target_size[0]),
                                image.GetSpacing()[1] * (image.GetSize()[1] / target_size[1]),
                                image.GetSpacing()[2] * (image.GetSize()[2] / target_size[2])])
    resampler.SetInterpolator(sitk.sitkLinear)
    image_resampled = resampler.Execute(image)
    image_array = sitk.GetArrayFromImage(image_resampled)
    image_norm = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    return image_norm

# Streamlit Interface
st.title("3D U-Net for Brain Tumor Segmentation")
st.header("Segmentation Test Interface")

selected_subject = st.selectbox("Choose a patient", sorted(os.listdir(val_image_dir)))

if st.button("Segment"):
    # Load and preprocess images
    image_files = [os.path.join(val_image_dir, file) for file in os.listdir(val_image_dir) if selected_subject in file]
    images = np.stack([preprocess_image(file) for file in image_files], axis=0)

    # Convert to tensor and predict
    images_tensor = torch.from_numpy(images).unsqueeze(0).to(device).float()
    with torch.no_grad():
        output = model(images_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Load ground truth mask
    ground_truth_path = os.path.join(val_mask_dir, f"{selected_subject}_seg.nii")
    ground_truth = sitk.GetArrayFromImage(sitk.ReadImage(ground_truth_path, sitk.sitkFloat32))

    # Display results
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(ground_truth[int(ground_truth.shape[0]/2)], cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[1].imshow(pred_mask[int(pred_mask.shape[0]/2)], cmap='gray')
    axs[1].set_title('Predicted Segmentation')
    st.pyplot(fig)

st.sidebar.write("Navigate through the subjects to display their brain tumor segmentations.")









