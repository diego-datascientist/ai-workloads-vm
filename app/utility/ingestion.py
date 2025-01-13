import time
import tracemalloc
import psutil
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from google.cloud import storage
import os
from dotenv import load_dotenv
import logging
from functools import wraps
import csv
import warnings
from urllib3.exceptions import InsecureRequestWarning

from helper_functions.embedding import ingestion
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

gcp_credentials_path = "./gcp-vm-key.json"
gcp_bucket_name = os.getenv('GCP_BUCKET_NAME')
gcp_folder = os.getenv('GCP_FOLDER_NAME')

if not os.path.exists(gcp_credentials_path):
    raise FileNotFoundError(f"GCP credentials file not found at {gcp_credentials_path}")

if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path

gcp_client = storage.Client()

############################################################################
# Global metrics dictionary
############################################################################
metrics = {
    "Download":    {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
    "Upload":        {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
    "Ingestion":  {"time": 0.0, "memory": 0.0, "cpu": 0.0, "bytes": 0.0, "calls": 0},
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
# Download
############################################################################
@benchmark(stage="Download")
def download_from_s3(s3_client, bucket, key, local_path=None):
    try:
        if not local_path:
            logger.error("download_from_s3 called without local_path!")
            return

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {key} to {local_path}")

        if os.path.isfile(local_path):
            local_path_sizes[local_path] = os.path.getsize(local_path)

    except Exception as e:
        logger.error(f"Error downloading {key} from S3: {e}")

@benchmark(stage="Download")
def download_from_MinIO(minio_client, bucket, key, local_path=None):
    try:
        if not local_path:
            logger.error("download_from_MinIO called without local_path!")
            return

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        minio_client.fget_object(bucket, key, local_path)
        logger.info(f"Downloaded '{key}' to '{local_path}'")

        if os.path.isfile(local_path):
            local_path_sizes[local_path] = os.path.getsize(local_path)

    except Exception as e:
        logger.error(f"Error downloading {key} from MinIO: {e}")

@benchmark(stage="Download")
def download_from_gcp(gcp_bucket_name=None, gcp_blob_name=None, local_path=None):
    try:
        if not local_path:
            logger.error("download_from_gcp called without local_path!")
            return

        bucket = gcp_client.bucket(gcp_bucket_name)
        blob = bucket.blob(gcp_blob_name)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {gcp_blob_name} to {local_path}")

        if os.path.isfile(local_path):
            local_path_sizes[local_path] = os.path.getsize(local_path)

    except Exception as e:
        logger.error(f"Error downloading {gcp_blob_name} from GCP: {e}")



############################################################################
# Upload
############################################################################
@benchmark(stage="Upload")
def upload_to_gcp(local_path=None, gcp_bucket_name=None, gcp_blob_name=None):
    try:
        if not local_path:
            logger.error("upload_to_gcp called without local_path!")
            return
        if not os.path.isfile(local_path):
            logger.warning(f"Cannot upload. Not a file: {local_path}")

        bucket = gcp_client.bucket(gcp_bucket_name)
        blob = bucket.blob(gcp_blob_name)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} to gs://{gcp_bucket_name}/{gcp_blob_name}")

    except Exception as e:
        logger.error(f"Error uploading {local_path} to GCP: {e}")

@benchmark(stage="Upload")
def upload_to_MinIO(minio_client, bucket, key, local_path=None):
    try:
        if not local_path:
            logger.error("upload_to_MinIO called without local_path!")
            return

        if not os.path.isfile(local_path):
            logger.warning(f"Cannot upload. Not a file: {local_path}")
            return

        with open(local_path, 'rb') as file_data:
            file_stat = os.stat(local_path)
            minio_client.put_object(
                bucket_name=bucket,
                object_name=key,
                data=file_data,
                length=file_stat.st_size
            )
        logger.info(f"Uploaded '{local_path}' to MinIO bucket '{bucket}' as '{key}'")
    except Exception as e:
        logger.error(f"Error uploading {local_path} to MinIO: {e}")

@benchmark(stage="Upload")
def upload_to_infinia(s3_client, bucket_name, infinia_folder_name, file_path):
    try:
        if not os.path.isfile(file_path):
            return f"Error: File '{file_path}' not found."

        file_name = os.path.basename(file_path)
        s3_key = os.path.join(infinia_folder_name, file_name)

        s3_client.upload_file(file_path, bucket_name, s3_key)
        return f"File '{file_name}' successfully uploaded to bucket '{bucket_name}' in folder '{infinia_folder_name}'."
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except NoCredentialsError:
        return "Error: No credentials provided."
    except PartialCredentialsError:
        return "Error: Incomplete credentials provided."
    except Exception as e:
        return f"An error occurred: {e}"


############################################################################
# Ingest
############################################################################
@benchmark(stage="Ingestion")
def ingest_data(file_name=None, local_path=None):
    try:
        ingestion(file_name, local_path)
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")


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


def ingestion_from_AWS(local_download_path):
    aws_bucket = input("Enter the AWS Bucket Name: ")
    aws_folder = input("Enter the AWS Folder Name: ")

    source_cred = {
        "accessKey": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "secretAccessKey": os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    }
    s3_client = boto3.client(
        's3',
        aws_access_key_id=source_cred['accessKey'],
        aws_secret_access_key=source_cred['secretAccessKey']
    )

    s3_files = list_files_in_s3(s3_client, aws_bucket, aws_folder)
    if not s3_files:
        logger.info(f"No files found in {aws_bucket}/{aws_folder}")
        return

    for key, size in s3_files:
        file_name = key.split('/')[-1]
        local_path = os.path.join(local_download_path, file_name)
        local_path_sizes[local_path] = size

        download_from_s3(s3_client, bucket=aws_bucket, key=key, local_path=local_path)
        upload_to_gcp(local_path=local_path, gcp_bucket_name=gcp_bucket_name, gcp_blob_name=(gcp_folder + file_name))
        ingest_data(file_name=file_name, local_path=local_path)
        remove_file(local_path)

def ingestion_from_infinia(local_download_path):
    infinia_bucket = input("Enter the Infinia Bucket Name:")
    infinia_folder = input("Enter the Infinia Folder Name: ")
    infinia_destination_bucket = "rag-lake"
    infinia_destination_folder = infinia_folder

    source_cred = {
        "accessKey": os.environ.get("INFINIA_ACCESS_KEY_ID", ""),
        "secretAccessKey": os.environ.get("INFINIA_SECRET_ACCESS_KEY", ""),
        "infinia_endpoint": os.environ.get("INFINIA_ENDPOINT_URL", "")
    }
    infinia_client = boto3.client(
        's3',
        aws_access_key_id=source_cred['accessKey'],
        aws_secret_access_key=source_cred['secretAccessKey'],
        endpoint_url=source_cred['infinia_endpoint'],
        verify=False
    )

    infinia_files = list_files_in_s3(infinia_client, infinia_bucket, infinia_folder)
    if not infinia_files:
        logger.info(f"No files found in {infinia_bucket}/{infinia_folder}")
        return
    for key, size in infinia_files:
        file_name = key.split('/')[-1]
        local_path = os.path.join(local_download_path, file_name)
        local_path_sizes[local_path] = size

        download_from_s3(infinia_client, bucket=infinia_bucket, key=key, local_path=local_path)
        upload_to_infinia(infinia_client, infinia_destination_bucket, infinia_destination_folder, local_path)
        ingest_data(file_name=file_name, local_path=local_path)
        remove_file(local_path)

def ingestion_from_MinIO(local_download_path):
    minio_source_bucket = input("Enter the MinIO Bucket Name: ")
    minio_source_folder = input("Enter the MinIO Folder Name (prefix): ")

    destination_minio_bucket = "gke-rag-destination-bucket"
    destination_prefix = minio_source_folder

    MINIO_SERVER = os.environ.get("MINIO_SERVER", "127.0.0.1:9000")
    MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

    minio_client = Minio(
        MINIO_SERVER,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    minio_files = minio_client.list_objects(
        bucket_name=minio_source_bucket,
        prefix=minio_source_folder,
        recursive=True
    )

    if not minio_files:
        logger.info(f"No files found in {minio_source_bucket}/{minio_source_folder}")
        return

    for file in minio_files:
        object_name = file.object_name
        file_basename = os.path.basename(object_name)
        local_path = os.path.join(local_download_path, file_basename)

        download_from_MinIO(minio_client, bucket=minio_source_bucket, key=object_name, local_path=local_path)
        destination_key = os.path.join(destination_prefix, file_basename)
        upload_to_MinIO(minio_client, bucket=destination_minio_bucket, key=destination_key, local_path=local_path)
        ingest_data(file_name=file_basename, local_path=local_path)
        remove_file(local_path)

def ingestion_from_GCP(local_download_path):
    destination_bucket_name = "gke-rag-destination-bucket"
    source_bucket_name = input("Enter the GCP Bucket Name: ")
    source_prefix = input("Enter the GCP Folder Name (prefix): ")

    source_bucket = gcp_client.bucket(source_bucket_name)
    blobs = list(source_bucket.list_blobs(prefix=source_prefix))

    if not blobs:
        logger.info(f"No files found in gs://{source_bucket_name}/{source_prefix}")
        return

    for blob in blobs:
        if os.path.isdir(blob.name):
            continue
        file_name = blob.name.split('/')[-1]
        local_path = os.path.join(local_download_path, file_name)

        download_from_gcp(gcp_bucket_name=source_bucket_name, gcp_blob_name=blob.name, local_path=local_path)
        upload_to_gcp(local_path=local_path, gcp_bucket_name=destination_bucket_name, gcp_blob_name=file_name)
        ingest_data(file_name=file_name, local_path=local_path)
        remove_file(local_path)


############################################################################
# Print & Save Metrics
############################################################################
def print_aggregate_metrics(aws):
    logger.info("\n======= Aggregate Metrics =======")
    csv_file = "final_metrics.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        if aws:
            writer.writerow(["Statistics of AWS, GCP, and Milvus"])
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
        else:
            writer.writerow(["Statistics of MinIO, GCP, and Milvus"])
            writer.writerow([
                "Stage", 
                "ExecutionTime(sec)",
                "MemoryUsage(MB)",
                "CPU-Usage(%)",
                "DataTransferred(MB)",
                "Throughput(MB/s)",
                "ModelName",
                "EmbeddingModelName",
                "Re-rankingModelName"
            ])
            for stage, data in metrics.items():
                print("Stage: ", stage)
                if stage == "Ingestion":
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
                        f"TotalThroughput: {throughput:.2f} MB/s",
                        "ModelName: "
                    )

                    writer.writerow([
                        stage,
                        f"{total_time:.2f}",
                        f"{total_mem:.2f}",
                        f"{avg_cpu:.2f}",
                        f"{(total_bytes / (1024 * 1024)):.2f}",
                        f"{throughput:.2f}"
                    ])
                else:
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
# Transfer
############################################################################

def transfer(local_download_path):
    os.makedirs(local_download_path, exist_ok=True)

    print(
        "Select one from the following:\n"
        "1. Press 1 for AWS S3\n"
        "2. Press 2 for MinIO Bucket\n"
        "3. Press 3 for GCP Bucket\n"
        "4. Press 4 for Infinia Bucket"
    )
    option = int(input("Enter your option: "))
    if option == 1:
        ingestion_from_AWS(local_download_path)
        print_aggregate_metrics(True)
    elif option == 2:
        ingestion_from_MinIO(local_download_path)
        print_aggregate_metrics(False)
    elif option == 3:
        ingestion_from_GCP(local_download_path)
        print_aggregate_metrics(False)
    elif option == 4:
        ingestion_from_infinia(local_download_path)
        print_aggregate_metrics(True)
############################################################################
# Main
############################################################################
if __name__ == "__main__":
    download_dir = "/home/daguero/idp-vm/app/utility/input"
    transfer(download_dir)