from pyspark.sql import SparkSession
from google.cloud import storage
import io
import os
from PIL import Image
from pyspark.ml.image import ImageSchema
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pyspark import SparkContext
import logging
from urllib.parse import urlparse



# ------------------------
# 1. Logging Configuration
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 2. PySpark Initialization
# -------------------------
def initialize_spark(app_name="ImageNet_Processing"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config('spark.jars', '/usr/local/lib/gcs-connector-hadoop3-latest.jar') \
        .getOrCreate()
    logger.info("Spark Session initialized.")
    return spark

# ------------------------
# 3. Device Configuration
# ------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

# ------------------------
# 4. Configure GCS Access
# ------------------------
def gcs_access(spark):
    spark.conf.set('spark.hadoop.google.cloud.auth.service.account.enable', 'true')
    spark.conf.set('spark.hadoop.google.cloud.auth.service.account.json.keyfile', 'app/utility/gcp-vm-key.json')
    spark.conf.set('spark.hadoop.fs.gs.impl', 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem')
    spark.conf.set('spark.hadoop.fs.gs.auth.service.account.json.keyfile', 'app/utility/gcp-vm-key.json')
    return spark

# -------------------------
# 5. Initialize GCS client
# -------------------------
def gcs_client():
    client = storage.Client()
    return client

# ----------------------------
# 6. Download images from GCS
# ----------------------------
def download_images(client, bucket_name, folder_path, local_dir):
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    for blob in blobs:
        if blob.name.endswith('/'):
            continue
        
        local_file_path = os.path.join(local_dir, os.path.basename(blob.name))
        blob.download_to_filename(local_file_path)
        print(f"Downloaded: {blob.name} to {local_file_path}")


# ----------------------------
# 7. Load ImageNet Using PySpark
# ----------------------------
def load_images(spark, img_dir):
    try:
        img_df = spark.read.format("image").load(img_dir)
        logger.info(f"Loaded {img_df.count()} images from {img_dir}")
        return img_df
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        raise

# ----------------------------------
# 8. Custom PyTorch Dataset for ImageNet
# ----------------------------------
class ImageNetDataset(Dataset):
    def __init__(self, img_df, transform=None, base_dir='./'):
        self.img_df = img_df.collect()
        self.transform = transform
        self.base_dir = base_dir

    def __len__(self):
        return len(self.img_df)

    def __getitem__(self, idx):
        img_path = self.img_df[idx].image.origin
        parsed_path = urlparse(img_path).path  
        relative_path = os.path.relpath(parsed_path, self.base_dir)

        try:
            img = Image.open(relative_path).convert("RGB")

            filename = os.path.basename(relative_path)
            parts = filename.split('.')
            
            label_str = parts[1]
            label = int(label_str)

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.warning(f"Failed to process image: {relative_path}. Error: {e}")
            return None

# -----------------------------------------
# 9. Image Transformations and Dataloaders
# -----------------------------------------
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# ----------------------------------
# 10. Model Configuration (ResNet50)
# ----------------------------------
def get_model(device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)  # 1000 classes for ImageNet
    model = model.to(device)
    logger.info("Loaded pre-trained ResNet50 model.")
    return model

# ---------------------------
# 11. Training Configuration
# ---------------------------
def configure_training(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, optimizer

# -----------------------------------
# 12. Training Loop with Checkpoints
# -----------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    checkpoint_path = "./checkpoints/"
    os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            if images is None:
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            checkpoint_file = f"{checkpoint_path}/resnet50_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_file)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")

    logger.info("Training complete. Saving final model.")
    torch.save(model.state_dict(), "./final_model_resnet50.pth")

# ---------------------
# 13. Model Evaluation
# ---------------------
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Accuracy on validation set: {accuracy:.2f}%")


# ------------------------
# 14. Upload Checkpoints
# -----------------------
def upload_checkpoints(client, bucket_name, local_checkpoints_dir, gcs_folder="checkpoints"):
    bucket = client.bucket(bucket_name)

    try:
        for root, dirs, files in os.walk(local_checkpoints_dir):
            for filename in files:
                local_path = os.path.join(root, filename)

                relative_path = os.path.relpath(local_path, local_checkpoints_dir)
                gcs_blob_path = os.path.join(gcs_folder, relative_path)

                blob = bucket.blob(gcs_blob_path)
                blob.upload_from_filename(local_path)

                print(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_blob_path}")
    except Exception as e:
        logger.warning(f"Failed to upload checkpoints: {relative_path}. Error: {e}")


# ---------------------------
# 15. Main Pipeline Execution
# ---------------------------
if __name__ == "__main__":
    spark = initialize_spark()
    device = get_device()
    spark = gcs_access(spark)

    # CONSTANTS STARTS
    BUCKET_NAME = 'spark-sample-images'
    TRAIN_OBJ_NAME = 'train_data/'
    VAL_OBJ_NAME = 'val_data/'
    CHECKPOINTS_FOLDER = '/home/daguero/idp-vm/app/utility/checkpoints/'
    GCP_BUCKET_FOLDER_NAME = 'checkpoints'
    LOCAL_TRAIN_DIR = './app/utility/sample/train/'
    LOCAL_VAL_DIR = './app/utility/sample/val/'
    # CONSTANTS ENDS

    os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(LOCAL_VAL_DIR, exist_ok=True)

    client = gcs_client()
    download_images(client, BUCKET_NAME, TRAIN_OBJ_NAME, LOCAL_TRAIN_DIR)
    download_images(client, BUCKET_NAME, VAL_OBJ_NAME, LOCAL_VAL_DIR)

    train_df = load_images(spark, LOCAL_TRAIN_DIR)
    val_df = load_images(spark, LOCAL_VAL_DIR)

    transform = get_transforms()

    train_dataset = ImageNetDataset(train_df, transform)
    val_dataset = ImageNetDataset(val_df, transform)
    train_loader = create_dataloader(train_dataset)
    val_loader = create_dataloader(val_dataset, shuffle=False)

    model = get_model(device)
    criterion, optimizer = configure_training(model)

    train_model(model, train_loader, val_loader, criterion, optimizer, device)
    evaluate_model(model, val_loader, device)
    client = gcs_client()
    upload_checkpoints(client, BUCKET_NAME,CHECKPOINTS_FOLDER, GCP_BUCKET_FOLDER_NAME)
