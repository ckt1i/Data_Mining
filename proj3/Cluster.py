import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

class K_means:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize centroids randomly
        X = np.array(X)
        self.centroids = X[np.random.choice(range(X.shape[0]), self.k, replace=False)]

        for _ in range(self.max_iter):
            # Assign each data point to the closest centroid
            labels = np.argmin(np.linalg.norm(X[:, None] - self.centroids, axis=2), axis=1)

            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.labels = labels
        self.inertia = np.sum(np.linalg.norm(X[:, None] - self.centroids[labels], axis=2) ** 2)

class EM:
    def __init__(self, n_components, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X):
        # Initialize parameters randomly
        X = np.array(X)

        self.model = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter)
        self.model.fit(X)
        self.labels = self.model.predict(X)
        self.centroids = self.model.means_
        self.inertia = self.model.score(X)

    def predict(self, X):
        # Predict labels for new data
        X = np.array(X)
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(X)
        return labels

class SpectualClustering:
    def __init__(self, n_clusters, KN=10):
        self.n_clusters = n_clusters
        self.KN = KN

    def construct_similarity_matrix(self, X):
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.KN)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        similarity_matrix = np.exp(-distances ** 2 / (2 * (np.mean(distances) ** 2)))
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
        self.centroids = self.model.means_
        self.inertia = self.model.score(X)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        labels = self.model.predict(X)
        return labels

class Evaluation:
    def __init__(self, data, labels, centroids, new_data, new_labels):
        self.data = data
        self.labels = labels
        self.centroids = centroids
        self.new_data = new_data
        self.new_labels = new_labels

    def evaluate_NMI(self):
        from sklearn.metrics import normalized_mutual_info_score
        if len(self.labels) != len(self.new_labels):
            raise ValueError(f"Inconsistent number of samples: {len(self.labels)} and {len(self.new_labels)}")
        nmi = normalized_mutual_info_score(self.labels, self.new_labels)
        return nmi

    def evaluate_Silhouette(self):
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(self.data, self.labels)
        return silhouette


def visualize(data, labels, centroids, new_data, new_labels):
    # Visualize the data and the centroids
    data = np.array(data)
    centroids = np.array(centroids)
    new_data = np.array(new_data)

    # Using PCA to reduce the dimensionality of the data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    centroids_pca = pca.transform(centroids)
    new_data_pca = pca.transform(new_data)

    # Plotting the data and the centroids
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x')
    plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], c=new_labels, cmap='viridis', marker='x')
    plt.show()
    

# Example usage
def main():
    # Generate some random data
    np.random.seed(0)
    X = np.random.randn(100, 2)

   # fit Spectual Clustering
    sc = SpectualClustering(n_clusters=3)
    sc.fit(X)

    # Predict labels for new data
    new_labels = sc.predict(X)

    # Visualize results
    from matplotlib import pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=sc.labels, cmap='viridis')
    plt.scatter(sc.centroids[:, 0], sc.centroids[:, 1], c='red', marker='x')
    plt.scatter(X[:, 0], X[:, 1], c=new_labels, cmap='viridis', marker='x')
    plt.show()

    # Evaluate the model
    evaluation = Evaluation(X, sc.labels, sc.centroids, X, new_labels)
    print("NMI:", evaluation.evaluate_NMI())
    print("Silhouette:", evaluation.evaluate_Silhouette())

if __name__ == "__main__":
    main()