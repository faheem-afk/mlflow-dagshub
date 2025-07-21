from mlflow.models.signature import infer_signature
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv("student_performance_df.csv")

for i in df.columns:
    df[i] = df[i].astype('float')
    
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 3

mlflow.set_experiment('student-performance-dt')

with mlflow.start_run(run_name='decision-tree-experimentation'):

    dt = DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    signature = infer_signature(X_train, dt.predict(X_train))
    
    input_example = input_example = X_train.iloc[:5, :]
    
    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_param('max_depth', max_depth)

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0,1])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.title('Confusion Matrix')
    
    plt.savefig("Confusion_matrix.png")
    
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(dt, artifact_path='models', signature=signature, input_example=input_example)




