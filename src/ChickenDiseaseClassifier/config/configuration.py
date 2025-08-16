from ChickenDiseaseClassifier.constants import *
import os
from ChickenDiseaseClassifier.utils.common import read_yaml, create_directories
from ChickenDiseaseClassifier.entity.config_entity import (DataIngestionConfig,
                                                           AWSConfig, 
                                                           PrepareBaseModelConfig, 
                                                           PrepareCallbacksConfig,
                                                           TrainingConfig,
                                                           EvaluationConfig)


class ConfigurationManager:
    class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            bucket_name=config.bucket_name,
            file_key=config.file_key,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
    
    def get_aws_config(self) -> AWSConfig:
        if not hasattr(self.config, "aws") or self.config.aws is None:
            raise ValueError("AWS config not found. Add 'aws' section to config.yaml or skip S3 download.")
        aws = self.config.aws
        return AWSConfig(
            access_key_id=aws.access_key_id,
            secret_access_key=aws.secret_access_key,
            region_name=aws.region_name
        )
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),                # .keras
            updated_base_model_path=Path(config.updated_base_model_path),# .keras
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
    

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([Path(model_ckpt_dir), Path(config.tensorboard_root_log_dir)])
        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath) # .keras
        )
    

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Chicken-fecal-images")
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=int(params.EPOCHS),
            params_batch_size=int(params.BATCH_SIZE),
            params_is_augmentation=bool(params.AUGMENTATION),
            params_image_size=list(params.IMAGE_SIZE),
            learning_rate=float(params.LEARNING_RATE),
        )
    

    def get_validation_config(self) -> EvaluationConfig:
        # Align with training outputs and data dir
        return EvaluationConfig(
            path_of_model=Path("artifacts/training/model.keras"),
            training_data=Path("artifacts/data_ingestion/Chicken-fecal-images"),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )