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

    # Create Specclust object
    model = SpectualClustering(n_clusters=3, KN=int(np.sqrt(len(data))))

    # Fit the model
    model.fit(data)
    
    result_path = "proj3/results/D1_Specclust.txt"
    fig_path = "proj3/results/D1_Specclust.png"

    # Make evaluations and visualization
    evaluation_Specclust = Evaluation(data, model.labels, model.centroids, true_labels, result_path, fig_path)
#    evaluation_Specclust = Evaluation(data, model.labels, model.centroids, true_labels)
    evaluation_Specclust.evaluate()
    evaluation_Specclust.visualize()

if __name__ == '__main__':
    main()