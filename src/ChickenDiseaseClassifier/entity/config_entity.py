from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AWSConfig:
    access_key_id: str
    secret_access_key: str
    region_name: str

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    bucket_name: str
    file_key: str
    local_data_file: Path
    unzip_dir: Path

