import streamlit as st
import os
import config
import random
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib

# Title and Header



def Home():
    st.title("Brain Tumor Segmentation Using 3D FCNN")
    st.header("Group Number: Group 5")
    st.subheader("Team Members:")
    st.write("- Ujjawal")
    st.write("- Bala Krishna")
    st.write("- Bhagawath Sai")
    st.write("- Sai Avinash")

    # Project Relevance
    st.header("Project Relevance")
    st.write("""
    Brain tumor segmentation is a crucial task in medical imaging that assists in the diagnosis and treatment of patients. 
    Accurate segmentation helps radiologists and clinicians understand tumor regions, identify malignancies, and monitor treatment progress.
    The project aims to implement 3D Fully Convolutional Neural Networks (FCNNs) to segment various regions of brain tumors from multi-modal MRI scans.
    """)

    # Dataset Description
    st.header("Dataset Description")
    st.write("""
    This project utilizes the BraTS dataset, which contains pre-processed multi-modal MRI scans for brain tumor segmentation.
    The dataset includes the following modalities:
    - T1-weighted (T1)
    - T1-weighted post-contrast (T1Gd)
    - T2-weighted (T2)
    - Fluid-attenuated inversion recovery (FLAIR)
    
    Each modality provides unique insights into the tumor regions, which are essential for precise segmentation.
    """)

    # Types of Scans and Tumor Regions
    st.header("Types of Scans and Tumor Regions")
    st.write("""
    The dataset includes annotations for various tumor regions:
    - *Whole Tumor (WT)*: All visible tumor regions.
    - *Necrotic and Non-enhancing Tumor Core (NETC)*: Central dead tissue and non-enhancing regions.
    - *Enhancing Tumor (ET)*: Actively growing tumor cells.
    - *Surrounding Non-tumor Fluid Heterogeneity (SNFH)*: Swelling or fluid accumulation around the tumor.
    - *Resection Cavity*: Post-surgical void in the brain.
    
    Select a tumor region below to highlight its description.
    """)

    # Dropdown for Tumor Regions
    tumor_regions = {
        "Whole Tumor (WT)": "All visible tumor regions.",
        "Necrotic and Non-enhancing Tumor Core (NETC)": "Central dead tissue and non-enhancing regions.",
        "Enhancing Tumor (ET)": "Actively growing tumor cells.",
        "Surrounding Non-tumor Fluid Heterogeneity (SNFH)": "Swelling or fluid accumulation around the tumor.",
        "Resection Cavity": "Post-surgical void in the brain."
    }

    selected_region = st.radio("Select Tumor Region:", list(tumor_regions.keys()))

    # Highlighting Selected Region
    st.write(f"*Description of Selected Region:* {tumor_regions[selected_region]}")

    # Visualization Placeholder
    st.header("Visualization")
    st.write("""
    *Note:* Integration with MRI data will provide interactive visualization of selected tumor regions.
    For now, this is a placeholder for region highlighting.
    """)

def load_mri_data(patient_dir_path, selected_category):
    for file_name in os.listdir(patient_dir_path):
        if file_name.startswith(selected_category) and file_name.endswith('.nii'):
            file_path = os.path.join(patient_dir_path, file_name)
            mri_data = nib.load(file_path)
            return mri_data.get_fdata()
    return None
def visualization():

    subjects_dir = config.brats_2024_dir
    subjects = os.listdir(subjects_dir)

    patient_dir = random.choice(subjects)

    patient_dir_path=os.path.join(subjects_dir, patient_dir)
    patient_mri_scans = os.listdir(patient_dir_path)



    st.header("Image Viewer")

    categories = ['t1c','t1n','t2f','t2w','seg']
    view_categories = ['Axial', 'Sagittal', 'Coronal']

    # Streamlit UI for view category selection; default is 'Axial'
    selected_view_category = st.selectbox("Select a view category", options=view_categories,
                                          index=view_categories.index('Axial'))

    # Display MRI scans
    # Using columns to display images category-wise
    cols = st.columns(len(categories))
    for idx, category in enumerate(categories):
        with cols[idx]:
            st.subheader(f"Category: {category}")
            files_displayed = 0  # Counter to check if any files are displayed
            for file in os.listdir(patient_dir_path):
                pattern=f'-{category}.nii.gz'
                if file.endswith(pattern) and file.endswith('.nii.gz'):
                    file_path = os.path.join(patient_dir_path, file)
                    mri_data = nib.load(file_path).get_fdata()

                    if selected_view_category == 'Axial':
                        slice_selected = mri_data.shape[2] // 2
                        image = mri_data[:, :, slice_selected]
                    elif selected_view_category == 'Sagittal':
                        slice_selected = mri_data.shape[0] // 2
                        image = mri_data[slice_selected, :, :]
                    elif selected_view_category == 'Coronal':
                        slice_selected = mri_data.shape[1] // 2
                        image = mri_data[:, slice_selected, :]

                    # Display the image using matplotlib
                    fig, ax = plt.subplots()
                    ax.imshow(image, cmap='gray')
                    ax.axis('off')  # Hide axes
                    st.pyplot(fig)
                    files_displayed += 1

            if files_displayed == 0:
                st.write(f"No MRI data available for the category: {category}")


tabs = ["Home", "visualisation", "Model", "Results"]
tab = st.radio("Navigate", tabs)

Home()
visualization()


