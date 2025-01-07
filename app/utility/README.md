# ImageNet Processing with PySpark and PyTorch

This repository demonstrates a pipeline for processing ImageNet-like image data using **PySpark** and **PyTorch**, with data and checkpoints stored on **Google Cloud Storage (GCS)**.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Authentication & Configuration](#authentication--configuration)
5. [Project Structure](#project-structure)
6. [Script Walkthrough](#script-walkthrough)
7. [Usage](#usage)
8. [Key Functions & Components](#key-functions--components)  
   - [Logging Configuration](#logging-configuration)  
   - [PySpark Initialization](#pyspark-initialization)  
   - [Device Configuration](#device-configuration)  
   - [Configure GCS Access for Spark](#configure-gcs-access-for-spark)  
   - [GCS Client](#gcs-client)  
   - [Downloading Images from GCS](#downloading-images-from-gcs)  
   - [Loading Images into Spark](#loading-images-into-spark)  
   - [Custom PyTorch Dataset](#custom-pytorch-dataset)  
   - [Transforms & Dataloaders](#transforms--dataloaders)  
   - [ResNet50 Model Setup](#resnet50-model-setup)  
   - [Training Configuration](#training-configuration)  
   - [Training Loop & Checkpoints](#training-loop--checkpoints)  
   - [Model Evaluation](#model-evaluation)  
   - [Upload Checkpoints to GCS](#upload-checkpoints-to-gcs)  
9. [Extending & Customizing](#extending--customizing)

---

## Overview
This pipeline demonstrates how to use **PySpark** for large-scale data ingestion and **PyTorch** for deep learning on image data. It also shows how to integrate with **Google Cloud Storage (GCS)** to download training/validation data and upload model checkpoints.

---

## Prerequisites
1. **Python 3.7+**  
2. **PyTorch** (`torch` and `torchvision`)  
3. **PySpark** (with **Spark ML** module)  
4. **Google Cloud SDK** or a **service account key** with appropriate GCS permissions  
5. **gcs-connector-hadoop3-latest.jar** for Spark to access GCS

---

## Installation
1. **Clone or Copy** this repository.
2. **Install Requirements**:
   ```bash
   pip install pyspark google-cloud-storage torch torchvision
If you need GPU support, ensure you have CUDA installed and install the CUDA-enabled PyTorch build:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

3.	**Set Up the GCS Connector:** Place gcs-connector-hadoop3-latest.jar where Spark can access it, for example */usr/local/lib/*.


## Authentication & Configuration

•	**Option A (Service Account Key):**
	
1. Create a service account JSON key with Storage Object Admin or equivalent role on your GCS bucket.
2. Export it:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

•	**Option B (Application-Default Login):**
```bash
gcloud auth application-default login
```

## Project Structure
```bash
.
├── app
│   └── utility
│       ├── sample
│       │   ├── train
│       │   └── val
│       ├── gcp-vm-key.json
│       └── checkpoints
├── gcs-connector-hadoop3-latest.jar
├── main_script.py
├── requirements.txt
└── README.md
```
1. *app/utility/sample/train/* & *app/utility/sample/val/*: local image directories
2. *app/utility/checkpoints/*: where model checkpoints get saved locally
3. *gcp-vm-key.json*: placeholder for the GCS service account key
4. *main_script.py*: the main Python script containing the pipeline code

## Script Walkthrough
1.	**Initialize Spark** with GCS connector.
2.	**Set Device** to GPU (cuda) if available, otherwise CPU.
3.	**Configure Spark** for GCS (using your service account or default credentials).
4.	**Download Images** from GCS into local directories.
5.	**Load Images** into a Spark DataFrame for potential distributed processing.
6.	**Create a PyTorch Dataset** from the Spark DataFrame.
7.	**Apply Transforms** (resize, normalization) and create a DataLoader.
8.	**Initialize ResNet50** with a pretrained ImageNet weight.
9.	**Train the Model**, save checkpoints every 2 epochs.
10.	**Validate** the model on a validation set.
11.	**Upload Checkpoints** to GCS.

## Usage
1.	**Adjust Variables** in main_script.py:
```bash
BUCKET_NAME = 'your-gcs-bucket'
TRAIN_OBJ_NAME = 'train_data/'
VAL_OBJ_NAME = 'val_data/'
LOCAL_TRAIN_DIR = './app/utility/sample/train/'
LOCAL_VAL_DIR = './app/utility/sample/val/'
```

2.	**Run the Script:**
```bash
python main_script.py
```
- Downloads images from gs://your-gcs-bucket/train_data/ & gs://your-gcs-bucket/val_data/
- Trains ResNet50, saving checkpoints into ./app/utility/checkpoints/
- Finally, uploads these checkpoints to the specified GCS folder.

## Key Functions & Components
### Logging Configuration
```bash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```
- Initializes a logger for console output.

### PySpark Initialization
```bash
def initialize_spark(app_name="ImageNet_Processing"):
    # Creates a Spark session with the GCS connector jar
    return spark
```
- Used to read images with Spark’s image data source.

### Device Configuration
```bash
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
```
- Automatically chooses GPU if available.

### Configure GCS Access for Spark
```bash
def gcs_access(spark):
    # Sets spark.conf for GCS
    return spark
```
- Allows Spark to read/write from GCS using service account credentials or default credentials.

### GCS Client
```bash
def gcs_client():
    client = storage.Client()
    return client
```
- Python client for GCS operations (download/upload).

### Downloading Images from GCS
```bash
def download_images(client, bucket_name, folder_path, local_dir):
    # Lists blobs and saves each to local_dir
```
- Ensures you have local copies of training/validation images.

### Loading Images into Spark
```bash
def load_images(spark, img_dir):
    img_df = spark.read.format("image").load(img_dir)
    return img_df
```
- Loads images as a DataFrame with columns like *origin*, *height*, *width*, etc.

### Custom PyTorch Dataset
```bash
class ImageNetDataset(Dataset):
    ...
    def __getitem__(self, idx):
        # Opens image with PIL, derives label from filename
```
- Collects Spark DataFrame rows into a list, implementing PyTorch’s Dataset interface.

### Transforms and DataLoaders
```bash
def get_transforms():
    return transforms.Compose([...])

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=8):
    return DataLoader(dataset, ...)
```
- Resize to 224×224, normalize with ImageNet stats, and batch the data.

### ResNet50 Model Setup
```bash
def get_model(device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)
    return model.to(device)
```
- Loads a pretrained ResNet50 and assigns the final layer to 1000 classes.

### Training Configuration
```bash
def configure_training(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer
```
- Defines loss and optimizer.

### Training Loops and Checkpoints
```bash
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    # Saves .pth files in ./checkpoints/ every 2 epochs
```
- Standard PyTorch training loop with checkpointing.

### Model Evaluation
```bash
def evaluate_model(model, val_loader, device):
    # Calculates accuracy
```
- Runs inference on the validation set.

### Upload Checkpoints to GCS
```bash
def upload_checkpoints(client, bucket_name, local_checkpoints_dir, gcs_folder="checkpoints"):
    # Walks through local checkpoints folder and uploads each file to the GCS bucket
```
- Ensures your trained models are backed up to GCS.


## Extending & Customizing
1. **Number of Classes**: Change the last layer of ResNet50 if not using the standard 1000 ImageNet classes.
2. **File Naming Convention**: If you have different image naming, update parsing logic in ImageNetDataset.__getitem__.
3. **Hyperparameters**: Adjust batch_size, num_epochs, and learning_rate to suit your needs.
4. **Data Splits**: Use different folders or different partitioning logic for train/val/test sets.