import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(data)

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
print(X)
save_to_file('X.txt', str(X))
print(y)
save_to_file('Y.txt', str(y))