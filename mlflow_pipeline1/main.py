# Got from here: https://github.com/haythemtellili/Machine-learning-pipeline/blob/master/main.py
# https://dzlab.github.io/ml/2020/08/09/mlflow-pipelines/
# mlflow server --host 127.0.0.1 --port 8080
import logging
import traceback
import warnings

import mlflow
import click
import os


def workflow():
  with mlflow.start_run() as active_run:
    print("Launching 'download'")
    download_run = mlflow.run(".", "download", parameters={})
    download_run = mlflow.tracking.MlflowClient().get_run(download_run.run_id)

    print("Launching 'train'")
    train_run = mlflow.run(".", "train", parameters={})
    train_run = mlflow.tracking.MlflowClient().get_run(train_run.run_id)

if __name__ == '__main__':
    workflow()

# def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
#     """Launching new run for an entrypoint"""

#     print(
#         "Launching new run for entrypoint=%s and parameters=%s"
#         % (entrypoint, parameters)
#     )
#     submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
#     return submitted_run


# @click.command()
# def workflow():
#     """run the workflow"""
#     with mlflow.start_run(run_name="data-pipeline"):
#         mlflow.set_tag("mlflow.runName", "data-pipeline")
#         _run("download")
#         _run("train")


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")

#     try:
#         workflow()
#     except Exception as e:
#         print(f"Exception occured. Check logs. {e}")