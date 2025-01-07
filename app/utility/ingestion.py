import boto3
from google.cloud import storage
import os
from dotenv import load_dotenv
import logging

from helper_functions.embedding import ingestion , openai_ingestion

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# gcp_credentials_path = "/home/daguero/idp-vm/app/Ingestion-Utility/gcp-vm-key.json"
gcp_credentials_path = "./gcp-vm-key.json"
gcp_bucket_name = os.getenv('GCP_BUCKET_NAME')
gcp_folder = os.getenv('GCP_FOLDER_NAME')


# Ensure GCP credentials path exists
if not os.path.exists(gcp_credentials_path):
    raise FileNotFoundError(f"GCP credentials file not found at {gcp_credentials_path}")

# Set GCP credentials programmatically
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path


# Create GCP Storage client
gcp_client = storage.Client()


def list_files_in_s3(s3_client, bucket, prefix):
    """
    List all files in an S3 bucket folder (prefix).
    """
    objects = []
    continuation_token = None

    while True:
        list_params = {'Bucket': bucket, 'Prefix': prefix}
        if continuation_token:
            list_params['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**list_params)

        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append(obj['Key'])

        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break

    return objects


def download_from_s3(s3_client, bucket, key, download_path):
    """
    Download a file from S3 to local storage.
    """
    try:
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        # Download the file
        s3_client.download_file(bucket, key, download_path)
        print(f"Downloaded {key} to {download_path}")

    except Exception as e:
        print(f"Error downloading {key} from S3: {e}")


def upload_to_gcp(local_path, gcp_bucket_name, gcp_blob_name):
    """
    Upload a file to GCP bucket.
    """
    try:
        bucket = gcp_client.bucket(gcp_bucket_name)
        blob = bucket.blob(gcp_blob_name)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{gcp_bucket_name}/{gcp_blob_name}")

    except Exception as e:
        print(f"Error uploading {local_path} to GCP: {e}")


def remove_file(local_path):
         # Optional: Remove the file after upload to save space
        try:
            os.remove(local_path)
            print(f"Deleted {local_path}")
        except Exception as e:
            print(f"Failed to delete {local_path}: {e}")

def transfer(source_cred, source_bucket, source_folder, local_download_path):

    s3_client = boto3.client(
        's3',
        aws_access_key_id=source_cred['accessKey'],
        aws_secret_access_key=source_cred['secretAccessKey']
    )

    # List files in S3 folder
    files = list_files_in_s3(s3_client, source_bucket, source_folder)

    if not files:
        print(f"No files found in {source_bucket}/{source_folder}")
        return

    os.makedirs(local_download_path, exist_ok=True)

    # Transfer files
    for file_key in files:
        file_name = file_key.split('/')[-1]
        local_path = os.path.join(local_download_path, file_name)
        gcp_blob_name = f"{gcp_folder}{file_name}"
        # Download from S3
        download_from_s3(s3_client, source_bucket, file_key, local_path)
        # Range_read -> chunk the S3 files

        # Upload to GCP
        upload_to_gcp(local_path, gcp_bucket_name, gcp_blob_name)
        
        
        # Ingestion 
        # ingestion(file_name, local_path)
        print(file_name)
        ingestion(file_name, local_path)


        #remove 
        remove_file(local_path)


if __name__ == "__main__":
    
    download_dir="/home/daguero/idp-vm/app/utility/input"

    # aws_bucket_name = os.getenv('AWS_BUCKET_NAME')
    # aws_folder = os.getenv('AWS_FOLDER_NAME')
    aws_bucket_name = input("Enter the AWS Bucket Name: ")
    aws_folder = input("Enter the AWS Folder Name: ")
    cred = {
    "accessKey": os.environ.get("AWS_ACCESS_KEY_ID", ""),
    "secretAccessKey": os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    }
    transfer(cred, aws_bucket_name, aws_folder, download_dir)