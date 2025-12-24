import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
X_test = pd.read_csv("namadataset_preprocessing/X_test.csv")
y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").values.ravel()

# MLflow setup
mlflow.set_experiment("Telco Churn Classification - Basic")

with mlflow.start_run():
    mlflow.autolog()


    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    # Evaluation
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
