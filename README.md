# Venkey-Chicken-Disease-Classification

## Workflows
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml

## How to run

1. Install dependencies:

pip install -r requirements.txt

2. Run Pipeline:

dvc repro

Ensure artifacts/training/model.keras will be created.

3. Start the app:

python app.py

Open http://127.0.0.1:5000/ and upload an image.