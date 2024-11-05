from Cluster import *
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    # Read data from file
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    true_labels = []
    for line in lines:
        line = line.strip()
        if line:
            data.append(list(map(float, line.split(',')[0:-1])))
            true_labels.append(line.split(',')[-1])
            
    return data , true_labels


def main():
    # Read data
    data , true_labels = read_data('proj3/Datasets/iris.csv')

    # Create K_means object
    model = SpectualClustering(n_clusters=3)

    # Fit the model
    model.fit(data)
    
#    result_path = "proj3/results/D1_K_means.txt"
#    fig_path = "proj3/results/D1_K_means.png"

    # Make evaluations and visualization
#    evaluation_K_means = Evaluation(data, model.labels, model.centroids, true_labels, result_path, fig_path)
    evaluation_K_means = Evaluation(data, model.labels, model.centroids, true_labels)
    evaluation_K_means.evaluate()
    evaluation_K_means.visualize()

if __name__ == '__main__':
    main()