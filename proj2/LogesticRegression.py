from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = cm[0][0] / (cm[0][0] + cm[1][0])
    precesion = cm[0][0] / (cm[0][0] + cm[0][1])
    F1_Measure = 2 * precesion * cm[0][0] / (precesion + cm[0][0])
    return cm , accuracy , recall , precesion , F1_Measure

def draw_confusion_matrix(cm , modle_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'proj2/figs/CM_{modle_name}.png')
    plt.close(fig)

def draw_bar_chart(accuracy, recall, precision, F1_Measure, model_name):
    labels = ['accuracy', 'recall', 'precision', 'F1_Measure']
    values = [accuracy, recall, precision, F1_Measure]
    fig, ax = plt.subplots()  # 正确解包
    ax.bar(labels, values)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'{model_name} Performance Metrics')
    plt.savefig(f'proj2/figs/val_{model_name}.png')
    plt.close(fig)  # 关闭图形以释放内存

def draw_figure(cm , accuracy , recall , precesion , F1_Measure):
    draw_confusion_matrix(cm , "LogesticRegression")
    draw_bar_chart(accuracy , recall , precesion , F1_Measure , "LogesticRegression")

# Main function
def train(filepath):
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    cm , accuracy , recall , precesion , F1_Measure = evaluate_model(model, X_test, y_test)
    draw_figure(cm , accuracy , recall , precesion , F1_Measure)

if __name__ == '__main__':
    filepath = 'proj2/DataSets/breast+cancer+wisconsin+diagnostic/wdbc.csv'
    train(filepath)