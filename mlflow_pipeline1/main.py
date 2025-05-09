# Got from here: https://github.com/haythemtellili/Machine-learning-pipeline/blob/master/main.py
# https://dzlab.github.io/ml/2020/08/09/mlflow-pipelines/
# mlflow server --host 127.0.0.1 --port 8080
import mlflow
import logging
import traceback
import warnings
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def workflow():
    try:
        with mlflow.start_run() as active_run:
            print("Launching 'download'...")
            download_run = mlflow.run(".", entry_point="download", parameters={})
            download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)

            print("Launching 'train'...")
            train_run = mlflow.run(".", entry_point="train", parameters={})
            train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

    except Exception as e:
        logging.error("An error occurred during the workflow execution.")
        logging.error(traceback.format_exc())
        warnings.warn(f"Error: {str(e)}")

# Before launching extract all files: MLproject, conda.yaml, download.py, train.py to folder AI_homework
if __name__ == "__main__":
    workflow()
