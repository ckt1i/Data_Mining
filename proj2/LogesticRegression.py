from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the data
def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, 2:]  # 取每行从第三列开始的所有数据
    y = data.iloc[:,1]  # 取每行的第二列数据
    # Check lengths of X and y
    print(f'Length of X: {len(X)}')
    print(f'Length of y: {len(y)}')
    return X, y

# Divide the data into training and test sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Main function
def train(filepath):
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    filepath = 'proj2/DataSets/breast+cancer+wisconsin+diagnostic/wdbc.csv'
    train(filepath)