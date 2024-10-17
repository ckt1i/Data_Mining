from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def load_data(file_path , target_column):
    data = pd.read_csv(file_path) 
    y = data.iloc[:, target_column] # 取目标列的数据
    X = data.drop(columns=[data.columns[target_column]]) # 取除了目标列的所有数据
    # Check lengths of X and y
    print(f'Length of X: {len(X)}')
    print(f'Length of y: {len(y)}')
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model_DecTree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_model_NaiveBayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def train(file_path, target_column):
    print("start")
    X, y = load_data(file_path , target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_DecTree = train_model_DecTree(X_train, y_train)
    model_NaiveBayes = train_model_NaiveBayes(X_train, y_train)
    print("Decision Tree:")
    print(evaluate_model(model_DecTree, X_test, y_test))
    print("Naive Bayes:")
    print(evaluate_model(model_NaiveBayes, X_test, y_test))

if __name__ == '__main__':
    D1_path = ""
    D2_path = "proj2\DataSets\iris\iris.csv"
#    train(D1_path)
    train(D2_path , -1)