# Got from here: https://habr.com/ru/articles/737046/
from airflow.decorators import dag
from airflow.operators.python_operator import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

from utils import aggregate_predictions

# Define path to data
x_data_path = "/opt/airflow/data/X.txt"
y_data_path = "/opt/airflow/data/Y.txt"

# Define keyword arguments to use for all DockerOperator tasks
dockerops_kwargs = {
    "mount_tmp_dir": False,
    "mounts": [
        Mount(
            source="airflow-pipelines1/data", # Change to your path
            target="/opt/airflow/data/",
            type="bind",
        )
    ],
    "retries": 1,
    "api_version": "1.30",
    "docker_url": "tcp://docker-socket-proxy:2375", 
    "network_mode": "bridge",
}


# Create DAG
@dag("Innopolis", start_date=days_ago(0), schedule="@daily", catchup=False)
def taskflow():
    # Task 1
    load_data = DockerOperator(
        task_id="load_data",
        container_name="task__load_data,
        image="data-loader:latest",
        command=f"python3 data_load.py",
        **dockerops_kwargs,
    )

    # Task 2
    predict = DockerOperator(
        task_id="predict",
        container_name="task__predict",
        image="model-prediction:latest",
        command=f"python3 model_predict.py",
        **dockerops_kwargs,
    )

# При сборке контейнеров пока не разобрался с ошибкой: докер-компоуз не может скачать зависимости (не видит их как будто)
# Когда по отдельности собираю докерфайлы с помощью команды --network=host, то некоторые зависимости скачиваются
# Возможно какая-та сетевая проблема или в настройках docker-compose.
    load_data >> predict


taskflow()
