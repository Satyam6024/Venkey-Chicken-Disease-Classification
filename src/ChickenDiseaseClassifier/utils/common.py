import os
from box.exceptions import BoxValueError
import yaml
from ChickenDiseaseClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return a ConfigBox."""
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
        if content is None:
            # box will raise BoxValueError on None, but we can be explicit
            raise BoxValueError("Empty YAML")
        logger.info(f"yaml file: {path_to_yaml} loaded successfully")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Create a list of directories; logs each creation when verbose=True."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Save a dict as JSON to the given path."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load JSON from path and return as ConfigBox."""
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """Save binary object via joblib."""
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """Load binary object via joblib."""
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """Return size of a file as a human-friendly string in KB."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

def decodeImage(imgstring: str, fileName: str):
    """Decode base64 image string to file."""
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)

def encodeImageIntoBase64(croppedImagePath: str) -> bytes:
    """Encode a file to base64 bytes."""
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
