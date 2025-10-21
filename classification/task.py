import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./creditcard.csv')

print(df.info())
print(df['Class'].value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         random_state=42,
    stratify=y
)


model = LogisticRegression(random_state=42,  max_iter=1000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


model_balanced = LogisticRegression(class_weight='balanced',
                                    random_state=42,  max_iter=1000)

model_balanced.fit(X_train, y_train)

y_pred_balanced = model_balanced.predict(X_test)



print("Balanced Classification Report:")
print(classification_report(y_test, y_pred_balanced))


print("\n Balanced Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))