import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

data = pd.read_csv('telco-customer-churn-preprocessing.csv')

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Telco Customer Churn Prediction")

with mlflow.start_run() as run:
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)