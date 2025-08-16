from ChickenDiseaseClassifier.config.configuration import ConfigurationManager
from ChickenDiseaseClassifier.components.callbacks import PrepareCallback
from ChickenDiseaseClassifier.components.training import Training
from ChickenDiseaseClassifier import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        cfg_mgr = ConfigurationManager()

        # Prepare callbacks
        cb_cfg = cfg_mgr.get_prepare_callback_config()
        cb_prep = PrepareCallback(config=cb_cfg)
        callback_list = cb_prep.get_tb_ckpt_callbacks()

        # Training pipeline
        train_cfg = cfg_mgr.get_training_config()
        trainer = Training(config=train_cfg)
        trainer.get_base_model()
        trainer.train_valid_generator()
        trainer.train(callback_list=callback_list)


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        raise e
        