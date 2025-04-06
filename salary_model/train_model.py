# Import libraries and modules
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Training Data
df = pd.read_csv('Salary_predict.csv')
X = df[["experience", "age", "interview_score"]]
y = df[["Salary"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,random_state=0)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Import MLflow Module
import mlflow
# Run local Project
mlflow.projects.run(uri='./', entry_point='main', experiment_name='Salary Model')