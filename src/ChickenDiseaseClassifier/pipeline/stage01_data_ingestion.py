from ChickenDiseaseClassifier.config.configuration import ConfigurationManager
from ChickenDiseaseClassifier.components.data_ingestion import DataIngestion
from ChickenDiseaseClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cfg = ConfigurationManager()
        di_cfg = cfg.get_data_ingestion_config()

    # Option A: Local file already present at data_ingestion.local_data_file
    # Ensure artifacts/data_ingestion/data.zip exists before running.
        data_ingestion = DataIngestion(config=di_cfg, aws_config=None)
    # Optionally: If using S3, uncomment the next two lines and ensure get_aws_config works.
    # aws_cfg = cfg.get_aws_config()
    # data_ingestion = DataIngestion(config=di_cfg, aws_config=aws_cfg); data_ingestion.download_file_from_s3()

        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
