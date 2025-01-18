import boto3
from botocore.client import Config
import time
import tracemalloc
import psutil
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from google.cloud import storage
import os
from dotenv import load_dotenv
import logging
from functools import wraps
import csv
import warnings
from urllib3.exceptions import InsecureRequestWarning

from helper_functions.embedding import indexing
from minio import Minio

# Suppress only InsecureRequestWarning
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

############################################################################
# Global metrics dictionary
############################################################################
metrics = {
    "Download":    {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
    "Upload":        {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
    "indexing":  {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
}

local_path_sizes = {}

############################################################################
# Decorator to benchmark each stage
############################################################################
def benchmark(stage):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Begin memory trace
            tracemalloc.start()

            process = psutil.Process(os.getpid())
            start_cpu = process.cpu_times()
            start_time = time.time()

            result = func(*args, **kwargs)

            elapsed_time = time.time() - start_time
            end_cpu = process.cpu_times()

            current_mem, _ = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            cpu_time_spent = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
            cpu_usage_call = 0.0
            if elapsed_time > 0:
                cpu_usage_call = (cpu_time_spent / elapsed_time) * 100.0

            metrics[stage]["time"]   += elapsed_time
            metrics[stage]["memory"] += (current_mem / (1024 * 1024))
            metrics[stage]["cpu"]    += cpu_usage_call
            metrics[stage]["calls"]  += 1

            local_path = kwargs.get("local_path", None)

            file_size = 0
            if local_path and os.path.isfile(local_path):
                file_size = os.path.getsize(local_path)
            elif local_path and local_path in local_path_sizes:
                file_size = local_path_sizes[local_path]

            metrics[stage]["bytes"] += file_size

            logger.info(f"{func.__name__} completed in {elapsed_time:.2f} sec.")
            logger.info(f"Memory Usage: {current_mem / (1024 * 1024):.2f} MB")
            logger.info(f"CPU Usage: {cpu_usage_call:.2f}%")
            logger.info(f"File size (this call): {file_size / (1024 * 1024):.2f} MB\n")
            return result
        return wrapper
    return decorator


############################################################################
# List files in S3
############################################################################
def list_files_in_s3(s3_client, bucket, prefix):
    objects = []
    continuation_token = None

    while True:
        list_params = {'Bucket': bucket, 'Prefix': prefix}
        if continuation_token:
            list_params['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**list_params)

        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                size = obj['Size']  # bytes
                objects.append((key, size))

        if response.get('IsTruncated'):
            continuation_token = response.get('NextContinuationToken')
        else:
            break

    return objects

############################################################################
# S3 Client Configuration
############################################################################
def get_s3_client(endpoint=None, access_key=None, secret_key=None, region=None, verify=None):
    return boto3.client(
        's3',
        endpoint_url=endpoint,  # Set for MinIO; None for AWS
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,  # Required for AWS; ignored for MinIO
        config=Config(signature_version='s3v4'),
        verify=verify
    )

############################################################################
# Download
############################################################################
@benchmark(stage="Download")
def download_file(s3_client, bucket, object_name, local_path=None):
    s3_client.download_file(bucket, object_name, local_path)
    logger.info(f"Downloaded {object_name} from {bucket} to {local_path}")

############################################################################
# Upload
############################################################################
@benchmark(stage="Upload")
def upload_file(s3_client, bucket, object_name, local_path=None):
    object_name = object_name or local_path
    s3_client.upload_file(local_path, bucket, object_name)
    logger.info(f"Uploaded {local_path} to {bucket}/{object_name}")


############################################################################
# Index
############################################################################
@benchmark(stage="indexing")
def ingest_data(file_name=None, local_path=None):
    try:
        indexing(file_name, local_path)
    except Exception as e:
        logger.error(f"Error during indexing: {e}")


############################################################################
# Remove
############################################################################
def remove_file(local_path):
    try:
        if os.path.isfile(local_path):
            os.remove(local_path)
            logger.info(f"Deleted {local_path}")
    except Exception as e:
        logger.error(f"Failed to delete {local_path}: {e}")


############################################################################
# Print & Save Metrics
############################################################################
def print_aggregate_metrics(aws, filename):
    logger.info("\n======= Aggregate Metrics =======")
    csv_file = "final_metrics.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        if aws:
            writer.writerow([f"{filename}"])
            writer.writerow([
                "Stage", 
                "ExecutionTime(sec)",
                "MemoryUsage(MB)",
                "CPU-Usage(%)",
                "DataTransferred(MB)",
                "Throughput(MB/s)"
            ])
            for stage, data in metrics.items():
                total_time  = data["time"]
                total_mem   = data["memory"]
                total_cpu   = data["cpu"]
                calls       = data["calls"]
                total_bytes = data["bytes"]

                avg_cpu = (total_cpu / calls) if calls else 0.0

                throughput = 0.0
                if total_time > 0 and total_bytes > 0:
                    throughput = (total_bytes / (1024 * 1024)) / total_time

                logger.info(
                    f"{stage} - "
                    f"ExecutionTime: {total_time:.2f} sec, "
                    f"MemoryUsage: {total_mem:.2f} MB, "
                    f"CPU-Usage: {avg_cpu:.2f}%, "
                    f"DataTransferred: {total_bytes / (1024 * 1024):.2f} MB, "
                    f"TotalThroughput: {throughput:.2f} MB/s"
                )

                writer.writerow([
                    stage,
                    f"{total_time:.2f}",
                    f"{total_mem:.2f}",
                    f"{avg_cpu:.2f}",
                    f"{(total_bytes / (1024 * 1024)):.2f}",
                    f"{throughput:.2f}"
                ])


############################################################################
# indexing
############################################################################
def indexing_in_Milvus(client, source_bucket, source_folder, local_download_path, destination_bucket, destination_folder):
    s3_files = list_files_in_s3(client, source_bucket, source_folder)

    if not s3_files:
        logger.info(f"No files found in {source_bucket}/{source_folder}")
        return

    for key, size in s3_files:
        file_name = key.split('/')[-1]
        local_path = os.path.join(local_download_path, file_name)
        local_path_sizes[local_path] = size
        destination_object = f"{destination_folder}/{file_name}"

        download_file(client, source_bucket, key, local_path=local_path)
        # upload_file(client, destination_bucket, destination_object, local_path=local_path)
        ingest_data(file_name=file_name, local_path=local_path)
        print_aggregate_metrics(True, file_name)
        remove_file(local_path)


############################################################################
# Transfer
############################################################################

def transfer(local_download_path):
    os.makedirs(local_download_path, exist_ok=True)

    print(
        "Select one from the following:\n"
        "1. Press 1 for AWS S3\n"
        "2. Press 2 for MinIO Bucket\n"
        "3. Press 3 for Infinia Bucket"
    )
    option = int(input("Enter your option: "))
    if option == 1:
        aws_bucket = input("Enter the AWS Bucket Name: ")
        aws_folder = input("Enter the AWS Folder Name: ")
        aws_destination_bucket = "gke-rag-destination-bucket"
        aws_destination_folder = "sample"
        source_cred = {
            "accessKey": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "secretAccessKey": os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        }

        aws_client = get_s3_client(None, source_cred['accessKey'], source_cred['secretAccessKey'], region=None)
        indexing_in_Milvus(aws_client, aws_bucket, aws_folder, local_download_path, aws_destination_bucket, aws_destination_folder)
        # print_aggregate_metrics(True)
    elif option == 2:
        # certificate = "/home/daguero/idp-vm/app/utility/certificates/minio-certificate.crt"
        minio_source_bucket = input("Enter the MinIO Bucket Name: ")
        minio_source_folder = input("Enter the MinIO Folder Name (prefix): ")
        minio_destination_bucket = "gke-rag-destination-bucket"
        minio_destination_folder = minio_source_folder
        MINIO_SERVER = os.environ.get("MINIO_SERVER", "127.0.0.1:9000")
        MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

        minio_client = get_s3_client(MINIO_SERVER, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, region=None)
        indexing_in_Milvus(minio_client, minio_source_bucket, minio_source_folder, local_download_path, minio_destination_bucket, minio_destination_folder)
        # print_aggregate_metrics(True)
    elif option == 3:
        infinia_source_bucket = input("Enter the Infinia Bucket Name:")
        infinia_source_folder = input("Enter the Infinia Folder Name: ")
        infinia_destination_bucket = "rag-lake"
        infinia_destination_folder = infinia_source_folder
        source_cred = {
            "accessKey": os.environ.get("INFINIA_ACCESS_KEY_ID", "XYZ"),
            "secretAccessKey": os.environ.get("INFINIA_SECRET_ACCESS_KEY", "XYZ"),
            "infinia_endpoint": os.environ.get("INFINIA_ENDPOINT_URL", "localhost:8111"),
            "verify": os.environ.get("INFINIA_CERT_PATH", "/certificate.crt")
        }
        infinia_client = get_s3_client(source_cred['infinia_endpoint'], source_cred['accessKey'], source_cred['secretAccessKey'], region=None, verify=source_cred['verify'])
        indexing_in_Milvus(infinia_client, infinia_source_bucket, infinia_source_folder, local_download_path, infinia_destination_bucket, infinia_destination_folder)
        # print_aggregate_metrics(True)


############################################################################
# Main
############################################################################
if __name__ == "__main__":
    download_dir = "/home/daguero/idp-vm/app/utility/input"
    transfer(download_dir)






















































# # Configuration
# def get_s3_client(endpoint=None, access_key=None, secret_key=None, region=None):
#     """
#     Create an S3 client compatible with both AWS and MinIO.
#     :param endpoint: Custom endpoint URL for MinIO (None for AWS).
#     :param access_key: Access key for authentication.
#     :param secret_key: Secret key for authentication.
#     :param region: AWS region (ignored for MinIO).
#     :return: Configured S3 client.
#     """
#     return boto3.client(
#         's3',
#         endpoint_url=endpoint,  # Set for MinIO; None for AWS
#         aws_access_key_id=access_key,
#         aws_secret_access_key=secret_key,
#         region_name=region,  # Required for AWS; ignored for MinIO
#         config=Config(signature_version='s3v4'),
#     )

# # Upload a file
# def upload_file(s3_client, bucket, file_name, object_name=None):
#     object_name = object_name or file_name
#     s3_client.upload_file(file_name, bucket, object_name)
#     print(f"Uploaded {file_name} to {bucket}/{object_name}")

# # Download a file
# def download_file(s3_client, bucket, object_name, file_name):
#     s3_client.download_file(bucket, object_name, file_name)
#     print(f"Downloaded {object_name} from {bucket} to {file_name}")

# # List objects in a bucket
# def list_objects(s3_client, bucket):
#     response = s3_client.list_objects_v2(Bucket=bucket)
#     if 'Contents' in response:
#         print(f"Objects in bucket '{bucket}':")
#         for obj in response['Contents']:
#             print(f" - {obj['Key']} (Size: {obj['Size']} bytes, Last Modified: {obj['LastModified']})")
#     else:
#         print(f"No objects found in bucket '{bucket}'.")

# # Example Usage
# if __name__ == "__main__":
#     # Replace with your details
#     use_minio = True  # Set to False for AWS S3
#     endpoint = "http://35.193.251.14:9000" if use_minio else None
#     access_key = "minioadmin"
#     secret_key = "minioadmin"
#     region = "your_region" if not use_minio else None
#     bucket_name = "gke-rag-source-bucket"

#     # Create S3 client
#     s3_client = get_s3_client(endpoint, access_key, secret_key, region)

#     # File operations
#     # upload_file(s3_client, bucket_name, "/home/daguero/idp-vm/app/ddn info/ddn_api.pdf", "sample/uploaded_file.txt")
#     # download_file(s3_client, bucket_name, "sample/uploaded_file.txt", "/home/daguero/idp-vm/app/ddn info/downloaded_file.txt")
#     list_objects(s3_client, bucket_name)


