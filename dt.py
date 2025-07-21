import os
from mlflow.models.signature import infer_signature
import mlflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub.init(repo_owner='faheem-afk', repo_name='mlflow-dagshub', mlflow=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

df = pd.read_csv("student_performance_df.csv")

for i in df.columns:
    df[i] = df[i].astype('float')
    
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 3
n_estimators = 100
# Set the tracking URI
mlflow.set_experiment('experiment_by_ballu_randomforest')

with mlflow.start_run(run_name='random-forest-experimentation'):

    # dt = DecisionTreeClassifier(max_depth=max_depth)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    signature = infer_signature(X_train, rf.predict(X_train))
    
    input_example = input_example = X_train.iloc[:5, :]
    
    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_param('max_depth', max_depth)

    mlflow.set_tag("author", "ballu")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0,1])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plt.title('Confusion Matrix')
    
    plt.savefig("Confusion_matrix.png")
    
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(rf, 'random_forest_with_dagshub2', signature=signature, input_example=input_example)




