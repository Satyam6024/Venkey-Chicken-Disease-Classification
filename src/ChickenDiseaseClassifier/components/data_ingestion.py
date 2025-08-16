import os
import boto3
import zipfile
from ChickenDiseaseClassifier import logger
from ChickenDiseaseClassifier.utils.common import get_size
from ChickenDiseaseClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig, aws_config=None):
        self.config = config
        self.aws_config = aws_config

    def download_file_from_s3(self):
        # Optional path if using S3; requires boto3.
        import boto3  # import only if used
        if not os.path.exists(self.config.local_data_file):
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_config.access_key_id,
                aws_secret_access_key=self.aws_config.secret_access_key,
                region_name=self.aws_config.region_name
            )
            s3_client.download_file(
                Bucket=self.config.bucket_name,
                Key=self.config.file_key,
                Filename=str(self.config.local_data_file)
            )
            logger.info(f"{self.config.local_data_file} downloaded from S3.")
        else:
            logger.info(f"File already exists: {self.config.local_data_file}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted files to: {unzip_path}")

