import os
import sys
import boto3
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_file_to_s3(file_path, s3_key=None, content_type=None):
    # Get AWS credentials from environment variables
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    
    if not all([aws_access_key, aws_secret_key, bucket_name]):
        logger.error("AWS credentials or bucket name not found in environment variables")
        return None
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    # If s3_key is not provided, use the filename
    if not s3_key:
        s3_key = f"uploads/{file_path.name}"
    
    # If content_type is not provided, guess based on file extension
    if not content_type:
        extension = file_path.suffix.lower()
        if extension == '.mp4':
            content_type = 'video/mp4'
        elif extension == '.png':
            content_type = 'image/png'
        elif extension == '.jpg' or extension == '.jpeg':
            content_type = 'image/jpeg'
        elif extension == '.gif':
            content_type = 'image/gif'
        else:
            content_type = 'application/octet-stream'
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        
        # Upload file
        with open(file_path, 'rb') as data:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type
            )
        
        # Generate the URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        logger.info(f"Upload successful! URL: {s3_url}")
        return s3_url
    
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return None

def upload_directory_to_s3(directory_path, s3_prefix=None):
    directory_path = Path(directory_path)
    if not directory_path.is_dir():
        logger.error(f"Directory not found: {directory_path}")
        return False
    
    # If s3_prefix is not provided, use the directory name
    if not s3_prefix:
        s3_prefix = f"uploads/{directory_path.name}"
    
    # Ensure s3_prefix ends with a slash
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    success_count = 0
    failure_count = 0
    
    for file_path in directory_path.glob('**/*'):
        if file_path.is_file():
            # Calculate relative path from the directory
            relative_path = file_path.relative_to(directory_path)
            s3_key = f"{s3_prefix}{relative_path}"
            
            # Upload file
            result = upload_file_to_s3(file_path, s3_key)
            if result:
                success_count += 1
            else:
                failure_count += 1
    
    logger.info(f"Directory upload completed. Success: {success_count}, Failures: {failure_count}")
    return success_count > 0 and failure_count == 0

def main():
    parser = argparse.ArgumentParser(description='Upload files to S3 bucket')
    parser.add_argument('path', help='File or directory to upload')
    parser.add_argument('--key', help='S3 key (destination path in bucket)')
    parser.add_argument('--content-type', help='Content type of the file')
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        result = upload_file_to_s3(path, args.key, args.content_type)
        if result:
            print(f"File uploaded successfully: {result}")
            return 0
        else:
            print("File upload failed")
            return 1
    
    elif path.is_dir():
        result = upload_directory_to_s3(path, args.key)
        if result:
            print("Directory uploaded successfully")
            return 0
        else:
            print("Directory upload failed")
            return 1
    
    else:
        print(f"Path not found: {path}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
