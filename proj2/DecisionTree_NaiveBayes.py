from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = cm[0][0] / (cm[0][0] + cm[1][0])
    precesion = cm[0][0] / (cm[0][0] + cm[0][1])
    F1_Measure = 2 * precesion * cm[0][0] / (precesion + cm[0][0])
    evaluations = [accuracy , recall , precesion , F1_Measure]
    return cm , evaluations

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

def draw_figure(cm_DT, cm_NB , eval_DT , eval_NB):
    draw_confusion_matrix(cm_DT , "NeuralNetwork")
    draw_confusion_matrix(cm_NB , "NB")
    
    labels = ['accuracy', 'recall', 'precesion', 'F1_Measure']
    values_DT = eval_DT
    values_NB = eval_NB
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values_DT, width, label='Decision Tree')
    rects2 = ax.bar(x + width/2, values_NB, width, label='Naive Bayes')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Decision Tree and Naive Bayes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig(f'proj2/figs/varcom_DTNB.png')
    

def train(file_path, target_column):
    print("start")
    X, y = load_data(file_path , target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_DecTree = train_model_DecTree(X_train, y_train)
    model_NaiveBayes = train_model_NaiveBayes(X_train, y_train)
    cm_DT , eval_DT = evaluate_model(model_DecTree, X_test, y_test)
    cm_NB , eval_NB = evaluate_model(model_NaiveBayes, X_test, y_test)
    draw_figure(cm_DT, cm_NB , eval_DT , eval_NB)

if __name__ == '__main__':
    D1_path = ""
    D2_path = "proj2\DataSets\iris\iris.csv"
#    train(D1_path)
    train(D2_path , -1)