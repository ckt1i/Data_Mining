from Cluster import *
import numpy as np

def read_data(filename):
    # Read data from file
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    true_labels = []
    for line in lines:
        line = line.strip()
        if line:
            data.append(list(map(float, line.split(',')[1:])))
            true_labels.append(line.split(',')[0])
            
    return data , true_labels

def main():
    # Read data
    data , true_labels = read_data('proj3/Datasets/wine.data')
    mode_1 = input("Please input the mode you want to use: 1 for K-means, 2 for EM, 3 for Spectual Clustering: ")
    mode_2 = input("Please input the mode you want to use: 1 for save the result, 2 for show the result: ")
    
    if mode_1 == '1':
       # Create K_means object
        model = K_means(n_clusters=3)
        result_path = "proj3/results/D2_Kmeans.txt"
        fig_path = "proj3/results/D2_Kmeans.png"

    elif mode_1 == '2':
        # Create EM object
        model = EM(n_clusters=3)
        result_path = "proj3/results/D2_EM.txt"
        fig_path = "proj3/results/D2_EM.png"

    elif mode_1 == '3':
        # Create Specclust object
        model = SpectualClustering(n_clusters=3, KN=int(np.sqrt(len(data))))
        result_path = "proj3/results/D2_Specclust.txt"
        fig_path = "proj3/results/D2_Specclust.png"

    # Fit the model
    model.fit(data)

    # Make evaluations and visualization
    if mode_2 == '1':
        evaluation = Evaluation(data, model.labels, model.centroids, true_labels, result_path, fig_path)
    elif mode_2 == '2':
        evaluation = Evaluation(data, model.labels, model.centroids, true_labels)
    evaluation.evaluate()
    evaluation.visualize(is_pca=False, show_centroids=False)

if __name__ == '__main__':
    main()