import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

class K_means:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize centroids randomly
        X = np.array(X)
        self.centroids = X[np.random.choice(range(X.shape[0]), self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the closest centroid
            labels = np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.labels = labels
  
            
class EM:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize parameters randomly
        X = np.array(X)

        self.model = GaussianMixture(n_components=self.n_clusters, max_iter=self.max_iter)
        self.model.fit(X)
        self.labels = self.model.predict(X)
        self.centroids = self.model.means_


class SpectualClustering:
    def __init__(self, n_clusters, KN=10):
        self.n_clusters = n_clusters
        self.KN = KN

    def construct_similarity_matrix(self, X):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.KN)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # Initialize a full similarity matrix with zeros
        similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
        
        # Fill the similarity matrix with the computed similarities
        for i in range(X.shape[0]):
            for j in range(self.KN):
                similarity_matrix[i, indices[i, j]] = np.exp(-distances[i, j] ** 2 / (2 * (np.mean(distances) ** 2)))
        
        return similarity_matrix
    

    def fit(self, X):
        '''
        X: input data
        self.KN: parameters for KNN
        n_clusters: the dimension in clustering
        '''
        X = np.array(X)
        num_features = X.shape[1]

        if self.KN is None:
            self.KN = int(np.sqrt(num_features))

        # Construct similarity matrix
        similarity_matrix = self.construct_similarity_matrix(X)

        # Construct degree matrix
        degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))

        # Compute Laplacian matrix
        laplacian_matrix = degree_matrix - similarity_matrix

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[0:self.n_clusters]]

        # Perform K-means clustering on the eigenvectors
        from sklearn.cluster import KMeans
        self.model = KMeans(n_clusters=self.n_clusters)
        clusters = self.model.fit(eigenvectors) # Assign labels to the data points
        self.labels = clusters.labels_
        '''
        谱聚类不好直接得到中心点，因为谱聚类是基于图的，不是基于距离的，所以没有中心点的概念。
        '''
        self.centroids = None
        

class Evaluation:
    def __init__(self, data, labels, centroids, true_labels=None , result_path = None, fig_path = None):
        self.data = data
        self.labels = labels
        self.centroids = centroids
        self.true_labels = true_labels
        self.result_path = result_path
        self.fig_path = fig_path

    def evaluate(self):
        from sklearn.metrics import normalized_mutual_info_score , silhouette_score
        if self.true_labels is not None:
            nmi = normalized_mutual_info_score(self.labels, self.true_labels)
        silhouette = silhouette_score(self.data, self.labels)

        if self.result_path is not None:
            with open(self.result_path, 'w') as file:
                if self.true_labels is not None:
                    file.write(f'NMI: {nmi}\n')
                file.write(f'Silhouette: {silhouette}\n')
        else:
            if self.true_labels is not None:
                print(f'NMI: {nmi}\n')
            print(f'Silhouette: {silhouette}\n')
       
    def visualize(self):
        # Visualize the data and the centroids
        data = np.array(self.data)

        # Using PCA to reduce the dimensionality of the data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        if self.centroids is not None:
            centroids_pca = pca.transform(self.centroids)

        # Plotting the data and the centroids
        if self.true_labels is not None:

            from sklearn.preprocessing import LabelEncoder
            # Encode the true labels
            label_encoder = LabelEncoder()
            true_labels_encoded = label_encoder.fit_transform(self.true_labels)
            
             # Define markers and colors
            markers = ['o', 's', '^' , 'D', 'x', '*', '+', 'p', 'h', 'v']
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

            # Plot the data points with shapes corresponding to true labels and colors corresponding to predicted labels
            for i, label in enumerate(np.unique(true_labels_encoded)):
                for j, pred_label in enumerate(np.unique(self.labels)):
                    mask = (true_labels_encoded == label) & (self.labels == pred_label)
                    plt.scatter(data_pca[mask, 0], data_pca[mask, 1],
                                marker=markers[i], color=colors[j])
            
             # Add legend for true labels (shapes)
            for i, label in enumerate(np.unique(true_labels_encoded)):
                plt.scatter([], [], marker=markers[i], color='black', label=f'True {label_encoder.inverse_transform([label])[0]}')

            # Add legend for predicted labels (colors)
            for j, pred_label in enumerate(np.unique(self.labels)):
                plt.scatter([], [], marker='o', color=colors[j], label=f'Pred {pred_label}')

        else:
            plt.scatter(data_pca[:, 0], data_pca[:, 1], c=self.labels, cmap='viridis')
        
        if self.centroids is not None:
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x')
        
        plt.title('Cluster Visualization with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        
        if self.fig_path is not None:
            plt.savefig(self.fig_path)
        else:
            plt.show()


# Example usage
def main():
    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)

   # fit Spectual Clustering
    model = K_means(n_clusters=3)

   # Predict labels for new data
    model.fit(X)

    # Visualize the data and the centroids
    evaluation = Evaluation(X, model.labels, model.centroids)
    evaluation.evaluate()
    evaluation.visualize()


if __name__ == "__main__":
    main()