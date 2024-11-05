from proj3.Cluster import K_means
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    # Read data from file
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            data.append(list(map(float, line.split(',')[0:-1])))
    return data


def visualize(data, labels, centroids, new_data, new_labels):
    # Visualize the data and the centroids
    data = np.array(data)
    centroids = np.array(centroids)
    new_data = np.array(new_data)

    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis') 
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.scatter(new_data[:, 0], new_data[:, 1], c=new_labels, cmap='viridis')
    plt.show()

def evaluate(data, labels, centroids):
    pass

def main():
    # Read data
    data = read_data('proj3/Datasets/iris.csv')

    # Create K_means object
    kmeans = K_means(k=3)

    # Fit the model
    kmeans.fit(data)

    # Predict labels for new data
    new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [4.7, 3.2, 1.3, 0.2]]
    new_data = np.array(new_data)
    new_labels = kmeans.predict(new_data)

    # Visualize results
    visualize(data, kmeans.labels, kmeans.centroids, new_data, new_labels)

if __name__ == '__main__':
    main()