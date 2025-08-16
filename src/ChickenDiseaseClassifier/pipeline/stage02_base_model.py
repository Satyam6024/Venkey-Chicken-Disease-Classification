from ChickenDiseaseClassifier.config.configuration import ConfigurationManager
from ChickenDiseaseClassifier.components.base_model import PrepareBaseModel
from ChickenDiseaseClassifier import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cfg = ConfigurationManager()
        pbm_cfg = cfg.get_prepare_base_model_config()
        pbm = PrepareBaseModel(config=pbm_cfg)
        pbm.get_base_model()
        pbm.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e