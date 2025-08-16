import tensorflow as tf
from pathlib import Path
from ChickenDiseaseClassifier.entity.config_entity import EvaluationConfig
from ChickenDiseaseClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        # Use a validation split; ensure consistency with how you trained
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30  # Choose a split for evaluation; independent of training split
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        # Load without compile state, then compile explicitly for evaluation
        return tf.keras.models.load_model(path, compile=False)

    def evaluation(self):
        # Load and compile for evaluation
        self.model = self.load_model(self.config.path_of_model)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # or use self.config.all_params.LEARNING_RATE
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Build validation generator
        self._valid_generator()

        # Evaluate; use self.model (not a stray global)
        self.score = self.model.evaluate(self.valid_generator, verbose=1)

    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)
        print("Saved scores.json:", scores)