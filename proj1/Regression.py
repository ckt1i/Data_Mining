import numpy as np
import matplotlib.pyplot as plt

import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge , RidgeCV
import random

from D1_generation import *
from D2_Preprocessing import *



# This is the part for the functions used for Polynomial regression
def divide_data(x, y, test_size):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=random.randint(0,65536))
    return train_x, test_x, train_y, test_y


def Poly_train_model(degree , x , y):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    model = pl.make_pipeline(sp.PolynomialFeatures(degree), lm.LinearRegression())
    model.fit(x, y)
    return model

def model_evaluation(model, x, y):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    y_pred = model.predict(x)
    mae = np.mean(np.abs(y_pred - y))
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    return mae, rmse

def evaluate_model(max_degree , train_x , train_y , test_x , test_y):
    MAE_single = []
    RMSE_single = []

    for i in range(1,max_degree+1):
        model = Poly_train_model(i, train_x, train_y)
        mae, rmse = model_evaluation(model, test_x, test_y)
        MAE_single.append(mae)
        RMSE_single.append(rmse)

    return MAE_single , RMSE_single

def plot_models(train_x , train_y , test_x , test_y , x_sin , y_sin):
    color_list = ['green', 'yellow' , 'black', 'orange', 'purple', 'pink', 'brown', 'cyan']

    plt.scatter(train_x, train_y, color='blue', label = "Train Data")
    plt.scatter(test_x, test_y, color='red', label = "Test Data")
    plt.plot(x_sin, y_sin, color='black', linestyle='--')

    for i in range(1,6):
        model = Poly_train_model(i, train_x, train_y)
        pred_y = model.predict(x_sin.reshape(-1, 1))
        plt.plot(x_sin, pred_y, label = "Model in Degree "+str(i) , color = color_list[i-1])
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-2,2)

    plt.title('Polynomial Regression')
    plt.legend()
#    plt.savefig('PolyReg.svg')
#    plt.savefig('PolyReg.png') 
    plt.show()
    plt.close()

def plot_evaluation(MAEs , RMSEs , sample_num):
    plt.bar([0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9], MAEs, color='blue', width=0.2)
    plt.bar([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1], RMSEs, color='red', width=0.2)
    
    plt.title("Model Evaluation Metrics")
    
    plt.xlabel('Degree')
    plt.ylabel('Error')
    
    plt.legend(['MAE', 'RMSE'])
    
#    plt.savefig('PolyEval_'+str(sample_num)+'.svg')
#    plt.savefig('PolyEval_'+str(sample_num)+'.png')
    
    plt.show()
    plt.close()


# This is the Part for the functions used for Ridge regression
def read_csv():
    # Read data from file
    # Return X_train, X_test, y_train, y_test
    with open('DataSets/D2.data') as f:
        lines = f.readlines()
        X = []
        Y = []
        for line in lines:
            data = line.split(',')
            if data[0] == "aveAllR":
                continue
            tmp = data[0]
            tmp = float(tmp)
            X.append([float(x) for x in data[:len(data)-1]])
            Y.append(float(data[-1]))

    return X, Y

def RegArg(X, Y) :
    # Changing the superparameter lambda to comfirm the regression arguements and 
    # return the best lambda

    coefs = []

    lambdas = np.logspace(-3, 3, 100) # 10^-3 to 10^3

    for lam in lambdas:
        clf = Ridge(alpha=lam)
        clf.fit(X, Y)
        coefs.append(clf.coef_)

    plt.figure(figsize=(10, 5))
    plt.plot(lambdas, coefs)
    plt.xscale('log')
    plt.xlabel('Regularization Coefficient (λ)')
    plt.ylabel('Coefficients')
    plt.title('Ridge Regularization Path')
    plt.grid(True)
    plt.show()

def OptimalLam(X, Y):
    # 定义一组 λ 值用于交叉验证
    lambdas = np.logspace(-4, 3, 100)  # 从 10^-4 到 10^2 共 100 个值

    # 使用 RidgeCV 模型进行交叉验证
    ridge_cv = RidgeCV(alphas=lambdas, store_cv_values=True)  # store_cv_values=True 可以存储交叉验证时的均方误差
    ridge_cv.fit(X, Y)  # 拟合数据集
    optimal_lambda = ridge_cv.alpha_  # 最优的正则化系数
    print(f"Optimal λ value through cross-validation: {optimal_lambda}")
    return optimal_lambda

def divide_datas(X , Y):
    # Divide the data into training and testing sets randomly 
    # With the ratio of 8:2 randomly
    # Return X_train, X_test, y_train, y_test

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for i in range(len(X)):
        if np.random.rand() < 0.8:
            X_train.append(X[i])
            y_train.append(Y[i])
        else:
            X_test.append(X[i])
            y_test.append(Y[i])
    
    return X_train, X_test, y_train, y_test

def Ridge_train_model(X_train, y_train, lambdas):
    # Train the model with the training set
    # Return the trained model
    clf = Ridge(alpha=lambdas)
    clf.fit(X_train, y_train)
    return clf

def test_model(model, X_test, y_test):
    # Test the model with the testing set
    # Return the MAE and RMSE of the model
    y_pred = model.predict(X_test)
    MAE = np.mean(np.abs(y_pred - y_test))
    RMSE = np.sqrt(np.mean((y_pred - y_test) ** 2))
    return MAE, RMSE
    
def draw_plot(MAEs , RMSEs):

    plt.figure(figsize=(10, 5))

    MAE_avr = np.mean(MAEs)
    RMSE_avr = np.mean(RMSEs)
    MAEs.append(MAE_avr)
    RMSEs.append(RMSE_avr)

    plt.bar([0.9, 1.9, 2.9, 3.9, 4.9 , 5.9], MAEs, label='MAE' , width=0.2)
    plt.bar([1.1, 2.1, 3.1, 4.1, 5.1 , 6.1], RMSEs, label='RMSE', width=0.2)

    plt.legend(['MAE', 'RMSE']) # 添加图例

    plt.xticks([1, 2, 3, 4, 5, 6], ['1', '2', '3', '4', '5', 'Average'])

    plt.ylabel("values")
    plt.title("Comparison of MAE and RMSE")
    plt.xlabel("Experiment ids")
    plt.show()
    


# This is the part for the main function of Polynomial Regression
x_sin , y_sin = np.array([]) , np.array([])
train_x , train_y = np.array([]) , np.array([])
test_x , test_y = np.array([]) , np.array([])
sample_num = 100
MAEs = []
RMSEs = []

x_sin , y_sin = np.array([]) , np.array([])
train_x , train_y = np.array([]) , np.array([])
test_x , test_y = np.array([]) , np.array([])
sample_num = 100
MAEs = []
RMSEs = []

def D1_main():
    global x_sin, y_sin, train_x, train_y, test_x, test_y, MAEs, RMSEs, sample_num
    x_sin , y_sin = generate_sinusoid(0, 10, 1000)
    x , y = sampling(int(1000/sample_num) , x_sin , y_sin , 0.1)
    write_to_file(x , y , 'DataSets/D1_'+str(sample_num)+'.data')
    
    for i in range(100):
        train_x, test_x, train_y, test_y = divide_data(x, y, 0.2)
        MAEs_single , RMSEs_single = evaluate_model(10,train_x, train_y, test_x, test_y)
        MAEs.append(MAEs_single)
        RMSEs.append(RMSEs_single)

    MAEs = np.mean(MAEs, axis=0)
    RMSEs = np.mean(RMSEs, axis=0)
    
    print("MAE: ", MAEs)
    print("RMSE: ", RMSEs)
       
    
    plot_models(train_x , train_y , test_x , test_y , x_sin , y_sin)
    plot_evaluation(MAEs , RMSEs , sample_num)



# This is the part for the main function of Ridge Regression
def D2_main():
    original_path = 'DataSets/facial-and-oral-temperature-data-from-a-large-set-of-human-subject-volunteers-1.0.0/FLIR_groups1and2.csv'
    new_path = 'DataSets/D2.data'
    
    extract_data(original_path, new_path)

    X , Y = read_csv()
    RegArg(X, Y)
    OptimalLam(X, Y)

    MAEs = []
    RMSEs = []

    X, Y = read_csv() # Read the data from the file

    lambdas = OptimalLam(X, Y) # Optimal λ through cross-validation

    for i in range(5):
        X_train, X_test, y_train, y_test = divide_datas(X , Y) # Divide the data into training and testing sets
        
        model = Ridge_train_model(X_train, y_train, lambdas) # Train the model with the training set

        MAE , RMSE = test_model(model, X_test, y_test) # Test the model with the testing set

        MAEs.append(MAE)
        RMSEs.append(RMSE)

    draw_plot(MAEs , RMSEs)


    

def main():
    mode = input("Please input the mode you want to run: \n1 for Polynomial Regression\n2 for Ridge Regression\n")
    if mode == '1':
        D1_main()
    elif mode == '2':
        D2_main()
    else:
        print("Invalid input")
        exit()

if __name__ == '__main__':
    main()