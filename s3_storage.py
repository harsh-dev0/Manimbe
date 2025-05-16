import os
import boto3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class S3Storage:
    def __init__(self):
        self.is_enabled = False
        self.s3_client = None
        self.bucket_name = None
        
        try:
            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            self.bucket_name = os.environ.get("S3_BUCKET_NAME")
            aws_region = os.environ.get("AWS_REGION", "us-east-1")
            
            if aws_access_key and aws_secret_key and self.bucket_name:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region
                )
                self.is_enabled = True
                logger.info(f"S3 storage initialized with bucket: {self.bucket_name}")
            else:
                logger.warning("S3 storage not configured, using local storage")
        except Exception as e:
            logger.error(f"Error initializing S3 storage: {e}")
    
    def upload_file(self, local_file_path, s3_key):
        if not self.is_enabled:
            logger.info(f"S3 storage not enabled, skipping upload of {local_file_path}")
            return None
            
        try:
            self.s3_client.upload_file(
                local_file_path,
                self.bucket_name,
                s3_key
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
            logger.info(f"Uploaded {local_file_path} to S3: {url}")
            return url
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            return None
            
    def get_url(self, s3_key):
        if not self.is_enabled:
            return None
            
        return f"https://{self.bucket_name}.s3.amazonaws.com/{s3_key}"
