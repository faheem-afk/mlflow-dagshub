import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



df = pd.read_csv("student_performance_df.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 100
max_depth = 5



with mlflow.start_run():

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    print('accuracy', accuracy_score(y_test, y_pred) )




