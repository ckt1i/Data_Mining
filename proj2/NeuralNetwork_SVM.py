from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix ,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_path , target_column):
    data = pd.read_csv(file_path)
    y = data.iloc[:, target_column]
    X = data.drop(columns=[data.columns[target_column]])
    print(f'Length of X: {len(X)}') 
    print(f'Length of y: {len(y)}')
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    y_train = np.array(y_train) 
    return X_train, X_test, y_train, y_test


def search_hyperparameters(model, X_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(50), (100), (200), (300), (400), (500)],
        'max_iter': [100,150,200,250,300,400,500]
    }
    cv = StratifiedKFold(n_splits=3)  # Reduce n_splits to 3
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    print("Best parameters: ", grid_search.best_params_)
    return grid_search.best_params_

def train_model_NeuralNetwork(X_train, y_train):
    hyparam = search_hyperparameters(MLPClassifier(), X_train, y_train)
    model = MLPClassifier(hidden_layer_sizes = hyparam['hidden_layer_sizes'], max_iter = hyparam['max_iter']) 
#    model = MLPClassifier(hidden_layer_sizes = (100), max_iter = 300)
    model.fit(X_train, y_train)
    return model


def train_model_SVM(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def calculate_item_matrix(cm):
    recall = np.zeros(len(cm))
    precession = np.zeros(len(cm))
    F1_Measure = np.zeros(len(cm))
    for i in range(len(cm)):
        recall[i] = cm[i, i] / np.sum(cm[:, i])
        precession[i] = cm[i, i] / np.sum(cm[i, :])
        F1_Measure[i] = 2 * recall[i] * precession[i] / (recall[i] + precession[i])
    return recall , precession , F1_Measure

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall , precesion , F1_Measure= calculate_item_matrix(cm)
    macro_F1 = np.mean(F1_Measure)
    micro_F1 = np.sum(cm.diagonal()) / np.sum(cm)
    accur_mean = np.mean(accuracy)
    recall_mean = np.mean(recall)
    precesion_mean = np.mean(precesion)
    evaluations = [accur_mean , recall_mean , precesion_mean , macro_F1 , micro_F1]
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
    plt.savefig(f'proj2/figs/{data}_CM_{modle_name}.png')

def draw_figure(cm_NN, cm_SVM , eval_NN , eval_SVM):
    draw_confusion_matrix(cm_NN , "NN")
    draw_confusion_matrix(cm_SVM , "SVM")
    
    labels = ['accuracy', 'recall', 'precesion', 'F1(macro)', 'F1(micro)']
    values_NN = eval_NN
    values_SVM = eval_SVM
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values_NN, width, label='Neural Network')
    rects2 = ax.bar(x + width/2, values_SVM, width, label='SVM')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Neural Network and SVM')
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

    plt.savefig(f'proj2/figs/{data}_NNSVM.png')
    
        

def train(file_path, target_column):
    X, y = read_file(file_path , target_column)
    if data == "D3":
        y = y.apply(lambda x: 0 if x <= 5 else (1 if x == 6 else 7))
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_NeuralNetwork = train_model_NeuralNetwork(X_train, y_train)
    model_SVM = train_model_SVM(X_train, y_train)
    cm_NN , eval_NN = evaluate_model(model_NeuralNetwork, X_test, y_test)
    cm_SVM , eval_SVM = evaluate_model(model_SVM, X_test, y_test)
    draw_figure(cm_NN, cm_SVM , eval_NN , eval_SVM)

data = "D3"

if __name__ == '__main__':
    D2_path = "proj2/DataSets/iris/iris.csv"
    D3_path = "proj2/DataSets/wine+quality/winequality-red.csv"

    if data == "D2":
        train(D2_path, -1)
    if data == "D3":
        train(D3_path, -1)

'''
Best parameters for D2:  {'hidden_layer_sizes': 50, 'max_iter': 250}
Best parameters for D3:  {'hidden_layer_sizes': 50, 'max_iter': 500}
'''
