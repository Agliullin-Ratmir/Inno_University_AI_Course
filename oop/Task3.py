# 1 https://github.com/mlflow/mlflow/blob/master/mlflow/models/model.py#L573
# Здесь представлено защищенное поле _signature, которое необходимо для каждой модели свое (исходя из описания).
# Поэтому логично что данное поле находится внутри класса и является защищенным.

# 2
from mlflow.pyfunc.model import PythonModel, ChatModel
from mlflow.tensorflow import MlflowCallback
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext

callBack = MlflowCallback()
print(type(callBack)) # <class 'mlflow.tensorflow.callback.MlflowCallback'>
print(issubclass(ChatModel, PythonModel)) # True
data_bricks = DatabricksClusterRunContext()
print(isinstance(data_bricks, RunContextProvider)) # True

# 3 Примеры классов с разными телами методов, но одинаковыми названиями
# https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/databricks_cluster_context.py
# https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/databricks_command_context.py
# оба класса наследуются от https://github.com/mlflow/mlflow/blob/master/mlflow/tracking/context/abstract_context.py