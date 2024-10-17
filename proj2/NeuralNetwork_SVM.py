from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

def read_file(file_path , target_column):
    data = pd.read_csv(file_path)
    y = data.iloc[:, target_column]
    X = data.drop(columns=[data.columns[target_column]])
    print(f'Length of X: {len(X)}') 
    print(f'Length of y: {len(y)}')
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model_NeuralNetwork(X_train, y_train):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    return model

def train_model_SVM(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

def train(file_path, target_column):
    X, y = read_file(file_path , target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_NeuralNetwork = train_model_NeuralNetwork(X_train, y_train)
    model_SVM = train_model_SVM(X_train, y_train)
    print("Neural Network:")
    print(evaluate_model(model_NeuralNetwork, X_test, y_test))
    print("SVM:")
    print(evaluate_model(model_SVM, X_test, y_test))

if __name__ == '__main__':
    D2_path = "proj2\DataSets\iris\iris.csv"
    D3_path = "proj2\DataSets\wine+quality\winequality-red.csv"

    train(D2_path , -1)
#    train(D3_path , -1)
