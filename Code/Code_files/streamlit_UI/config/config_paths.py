import os
current_path = os.path.dirname(os.path.abspath(__file__))




# Go up three levels to the DL_Project directory
project_dir = os.path.abspath(os.path.join(current_path, '../../../'))

brats_2024_dir = os.path.join(project_dir, 'training_data1_v2')

# Specify the train_data directory path
train_base_dir = os.path.join(project_dir, 'train_data')

train_images_dir=os.path.join(train_base_dir, 'train_images')
train_mask_dir=os.path.join(train_base_dir, 'train_masks')


val_base_dir=os.path.join(project_dir, 'validation_data')
val_images_dir=os.path.join(val_base_dir, 'validation_images')
val_mask_dir=os.path.join(val_base_dir, 'validation_masks')

models_stream_dir = os.path.join(project_dir, 'models', 'models_streamlit_test')

residual_model_path=f'{models_stream_dir}/resdUnet3d_best_model.pth'

unet_model_path=f'{models_stream_dir}/unet3d_best_model.pth'

streamlit_dir = os.path.abspath(os.path.join(current_path, '..'))

utils_path = os.path.join(streamlit_dir, 'utils')

metrics_resunet= f'{utils_path}/resd_metrics.csv'
unet_metrics = f'{utils_path}/unet_metrics.csv'


seg_image = f'{utils_path}/seg_image.png'
print(seg_image)
unet_image=f'{utils_path}/unet_arc_img.png'
print(unet_image)

