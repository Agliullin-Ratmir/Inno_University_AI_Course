import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load the Iris dataset
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
X, y = datasets.load_iris(return_X_y=True)
print(X)
np.savetxt("/opt/airflow/data/X.txt", X, delimiter=" ", fmt="%f")
print(y)
np.savetxt("/opt/airflow/data/Y.txt", y, delimiter=" ", fmt="%f")
