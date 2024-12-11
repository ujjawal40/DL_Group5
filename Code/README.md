<h1>Code Directory README</h1>

<h2>Overview</h2>
<p>This directory contains the source code and resources developed for the BRATS 2024 brain tumor segmentation project. Below, detailed instructions are provided on setup, model training, and application deployment, along with the folder structure at various stages of the project.</p>

<h3>Initial Directory Structure</h3>
<pre>
Code/
├── Code_files/
├── models/
└── trial-prototype-codes/
</pre>

<h3>Data Access and Preparation</h3>
<p>Data for this project is sourced from the BRATS 2024 competition. You can access it via any of the following link:</p>
<ul>
  <li>Direct Data Link: <a href="https://drive.google.com/file/d/1gIGtXB-e1DNmlDrPHl7hM9aSFHJ8zyzA/view?usp=drive_link">Google Drive Data</a>
    <span>: We have uploaded the dataset to Google One Drive for easy access.</span>
  </li>
  <li>Competition Link: <a href="https://www.synapse.org/Synapse:syn53708249/wiki/626323">BRATS 2024 Competition</a>
    <blockquote style="background-color: #f9f9f9; border-left: 10px solid #ccc; margin: 1.5em 10px; padding: 0.5em 10px;">
        <strong>Note:</strong> To access the data, click "Data Access/Download" and complete the following steps:
        <ol>
            <li>Agree to the terms and conditions of the data by registering for the challenge.</li>
            <li>Complete the BraTS 2024 Data Access form.</li>
            <li>Accept the invite to join the BRATS 2024 Data Access Team.</li>
            <li>Click on "Data Access/Download" to download the datasets.</li>
        </ol>
    </blockquote>
  </li>
</ul>

<h4>After Data Upload:</h4>
<p>We expect you to upload the data in the Code folder as shown:</p>
<pre>
Code/
├── Code_files/
├── models/
├── trial-prototype-codes/
└── training_data1_v2/
</pre>

<h3>Running Data Conversion</h3>
<p>Use the <code>data_conversion.py</code> script in the 'Code_files' folder to process the raw data:</p>
<pre>python3 data_conversion.py</pre>
<p>This script generates 'train_data' and 'validation_data' from 'training_data1_v2'. Expect this process to take 30-45 minutes.</p>

<h4>After Running data_conversion.py:</h4>
<p>The file structure will be organized as follows, with specific subfolders for images and masks within the 'train_data' and 'validation_data' directories:</p>
<pre>
Code/
├── Code_files/
├── models/
├── trial-prototype-codes/
├── training_data1_v2/
├── train_data/
│   ├── train_images/
│   └── train_masks/  <!-- Segmented masks for each subject -->
└── validation_data/
    ├── validation_images/
    └── validation_masks/  <!-- Segmented masks for validation subjects -->
</pre>


<h3>Folders and Their Contents</h3>

<h4>1. Code_files</h4>
<p>This folder includes scripts and notes directly related to the project's main objectives:</p>
<ul>
  <li><strong>data_consersion.py</strong> As discussed above.</li>
  <li><strong>model_3dunet.py</strong> and <strong>validation_unet3d.py</strong> for training and validating the UNet3D model.</li>
  <li><strong>model_3dRESN-unet.py</strong> and <strong>validation_resdUnet3d.py</strong> for training and validating the Residual UNet3D model.</li>
  <li><strong>streamlit_UI</strong> folder for housing codes, resources to run the streamlit app.</li>
</ul>

<pre>
Code_files/
├── data_conversion.py
│   <!-- Converts the original dataset into training and validation datasets -->
├── model_3dunet.py
│   <!-- Trains the UNet3D model -->
├── validation_unet3d.py
│   <!-- Validates the trained UNet3D model using saved parameters -->
├── model_3dRESN-unet.py
│   <!-- Trains the Residual UNet3D model -->
├── validation_resdUnet3d.py
│   <!-- Validates the trained Residual UNet3D model using saved parameters -->
└── streamlit_UI/
  
</pre>

<h4>2. Models</h4>
<p>The 'models' directory stores the model parameters developed and utilized throughout the project's lifecycle. It is organized to facilitate easy access to both pre-trained models for immediate testing and dynamically generated models during training phases.</p>

<h5>Initial Models Directory Structure:</h5>
<p>Initially, when the repository is cloned or set up, the 'models' directory contains a pre-configured subdirectory with the best-performing models from our training and validation experience:</p>
<pre>
models/
└── models_streamlit_test/
    <!-- Contains the best-performing UNet3D and Residual UNet3D models -->
</pre>

<h5>Description of the Models Directory:</h5>
<p>This directory is structured to support both the deployment of the application and the ongoing development and training of new models:</p>
<ul>
  <li><strong>models_streamlit_test</strong>: This folder holds pre-trained models that are considered the best-performing up to date. They are used primarily by the Streamlit application for demonstration purposes or initial testing.</li>
  <li><strong>models_by_train</strong>: This folder is dynamically created during the training process. It stores the model parameters for each epoch as well as the best model parameters obtained during training sessions. It allows for tracking of training progress and easy retrieval of the most effective models.</li>
</ul>

<h5>Final Models Directory Structure:</h5>
<p>After training, the 'models' directory expands to include a new subdirectory that stores the models generated during the training process:</p>
<pre>
models/
├── models_streamlit_test/
└── models_by_train/
    <!-- Generated during training; stores models from each epoch and the best models -->
</pre>

<p>This organized structure ensures that users can easily differentiate between stable, pre-trained models and those currently being tested and refined.</p>


<h4>3. Trial-Prototype-Codes</h4>
<p>Contains experimental scripts and prototypes. These files are for internal testing and do not impact the main application directly.</p>

<h4>Streamlit Application Setup</h4>
<p>To visualize and interact with the data and model's outputs, navigate to the 'streamlit_UI' folder within the 'Code_files' directory. Here, the Streamlit application is set up and ready for execution.</p>

<h5>Structure of the Streamlit_UI Directory:</h5>
<p>The 'streamlit_UI' directory contains the main script and additional support files necessary for running the Streamlit application:</p>
<pre>
streamlit_UI/
├── app.py                   <!-- Main Streamlit application script -->
├── model_summary_3dunet.txt <!-- UNet3D model summary -->
├── model_summary_resunet.txt<!-- ResUNet model summary -->
├── residual_model.py        <!-- Script for residual model operations -->
├── residual_unet_transformation.py <!-- Residual UNet transformation script -->
├── unet_model.py            <!-- Script for UNet model operations -->
├── unet_predicting_transformation.py <!-- UNet prediction transformation script -->
├── config/
│   └── config_paths.py      <!-- Manages dynamic paths for resources -->
└── utils/
    ├── resd_metrics.csv     <!-- CSV file with metrics for the Residual UNet -->
    ├── seg_image.png        <!-- Segmented image sample -->
    ├── unet_arc.jpg         <!-- UNet architecture image -->
    ├── unet_metrics.csv     <!-- CSV file with metrics for the UNet -->
</pre>

<h5>Description and Running the Streamlit App:</h5>
<p>The <code>app.py</code> file is the core of the Streamlit application, which integrates all functionalities and displays the results. The folder also includes:</p>
<ul>
  <li><strong>config</strong>: This folder contains the <code>config_paths.py</code> file, which helps in handling dynamic file paths throughout the application, ensuring flexibility and ease of configuration changes without needing to alter the code base.</li>
  <li><strong>utils</strong>: This folder houses supporting files like images and CSV files, which are crucial for loading various resources into the Streamlit app to enhance functionality and user interaction.</li>
</ul>

<p>To run the Streamlit application, navigate to the 'streamlit_UI' directory and execute the following command:</p>
<pre>streamlit run app.py</pre>
<p>This command will start the Streamlit server and open the application in your web browser, allowing you to interact with the models and view the segmentation results dynamically.</p>

