import numpy as np
import copy
import psutil
import os
from time import time
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def txt2array(txt_path, delimiter):
    data_list = []
    with open(txt_path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")
        data_split = line.split(delimiter)
        temp = list(map(float, data_split))
        data_list.append(temp)

    data_array = np.array(data_list)
    return data_array


def get_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))


# Square and sum each element of the row
def rowNorms(data):
    return np.einsum("ij,ij->i", data, data)


# The distance between each sample of x and y
def euclideanDistance(x, y):
    xx = rowNorms(x)[:, np.newaxis]  # To a column vector, so that every row of dot adds the same number
    yy = rowNorms(y)[np.newaxis, :]  # With xx in the same way
    dot = np.dot(x, np.transpose(y))
    res = xx + yy - 2 * dot
    return res


# The f norm of a matrix, when x is a vector, is the Euclidean norm
def squaredNornal(x):
    x = np.ravel(x)
    return np.dot(x, x)


def distTwoSample(x, y):
    tol = x-y
    return np.sqrt(np.einsum("i,i->", tol, tol))


# To determine which central point a sample point is near, the index of that central point is returned
# Let's say I have three centers, and I return 0,1,2
def closest_center(sample, centers):
    closest_i = 0
    closest_dist = float('inf')
    for i, c in enumerate(centers):
        # According to Euclidean distance judgment, select the category of the center point of the minimum distance
        distance = get_distance(sample, c)
        if distance < closest_dist:
            closest_i = i
            closest_dist = distance
    return closest_i


# Convert the output in Sklearn to clusters
def change(M, clustering_label):
    clusters = [[] for _ in range(M)]
    for sample_i, sample in enumerate(clustering_label):
        clusters[sample_i].append(sample)
    return clusters


# Get the kmean++ center
def get_kmeansplus_centers(n_cluster, data):
    n_samples, n_feature = data.shape
    center = np.zeros((n_cluster, n_feature))
    center[0] = data[np.random.randint(n_samples)]
    for i in range(1, n_cluster):
        # Calculate the distance from each sample point to the existing center point
        distance_to_centers = np.square(euclideanDistance(data, center[[i for i in range(i)]]))
        # Select the minimum data distance
        closed_distance = np.min(distance_to_centers, axis=1)
        # roulette
        denominator = closed_distance.sum()
        point = np.random.rand() * denominator  # The pointers of roulette
        be_choosed = np.searchsorted(np.cumsum(closed_distance), point)
        # Avoid selecting the last index and causing the index to cross the boundary
        be_choosed = min(be_choosed, n_samples - 1)
        center[i] = data[be_choosed]
    return center


# Define the process for building the cluster
# The content of each cluster is the index of the sample, that is, the sample index is clustered for convenient operation
def create_clusters(centers, n_cluster, data):
    clusters = [[] for _ in range(n_cluster)]
    for sample_i, sample in enumerate(data):
        # Divide the sample into the nearest category area
        center_i = closest_center(sample, centers)
        # An index for storing samples
        clusters[center_i].append(sample_i)
    return clusters


# The new center point is calculated according to the clustering result of the previous step
def calculate_centers(clusters, n_cluster, data):
    n_samples, n_features = data.shape
    centers = np.zeros((n_cluster, n_features))
    # Take the current mean of each class sample as the new center point
    for i, cluster in enumerate(clusters):  # Cluster is the index of each category
        new_center = np.mean(data[cluster], axis=0)  # Average by column
        centers[i] = new_center
    return centers


# Hierarchical Clustering Center
def hierarchical_centers(children_, labels_, M, k, data):
    n_samples, n_features = data.shape
    M_centers = np.zeros((M, n_features))
    for i, cluster in enumerate(children_):
        new_center = np.mean(data[cluster], axis=0)
        M_centers[i] = new_center

    label = [[] for _ in range(k)]
    for sample_i, sample in enumerate(labels_):
        label[sample].append(sample_i)

    n_samples, n_features = M_centers.shape
    k_center = np.zeros((k, n_features))
    for i, cluster in enumerate(label):
        new_center = np.mean(M_centers[cluster], axis=0)
        k_center[i] = new_center
    return k_center


# Get the cluster category to which each sample belongs
def get_cluster_labels(clusters, data):
    cat = np.zeros(np.shape(data)[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            cat[sample_i] = cluster_i
    return cat


# Obtain the initial cluster value
def get_preclusters_numbers(data):
    n_sample, n_features = data.shape
    if n_sample < 10:
        return n_sample
    else:
        while n_sample > 10:
            n_sample //= 2
        return n_sample


# elkan
def elkan_kmeans(k, data, max_iterations, tol, centers):
    n_samples, n_features = data.shape
    min_label = np.zeros(n_samples)
    for i in range(n_samples):
        min_label[i] = closest_center(data[i], centers)
    for _ in range(max_iterations):
        dist_center_half = euclideanDistance(centers, centers) / 2
        label = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            min_dist = distTwoSample(data[i], centers[int(min_label[i])])
            for j in range(1, k):
                if min_dist > dist_center_half[int(min_label[i]), j]:
                    this_dist = distTwoSample(data[i], centers[j])
                    if this_dist < min_dist:
                        min_dist = this_dist
                        min_label[i] = j
            label[i] = min_label[i]

        # Calculate the clustering center
        new_center = np.zeros((k, n_features))
        num_center = np.zeros(k)
        for r, v in enumerate(label):
            new_center[v] += data[r]
            num_center[v] += 1

        np.maximum(num_center, 1, out=num_center)  # Prevent the division by 0
        np.reciprocal(num_center, dtype=float, out=num_center)
        np.einsum("ij,i->ij", new_center, np.transpose(num_center), out=new_center)

        # Evaluate with the F norm
        center_shift_total = squaredNornal(centers - new_center)
        if squaredNornal(center_shift_total < tol):
            break
        centers = copy.deepcopy(new_center)
    clusters = create_clusters(centers, k, data)
    return clusters


# Define HTK algorithm flow according to the above flow
def HTK(data, threshold, tol, max_iterations):
    M = get_preclusters_numbers(data)
    # 1. Initialize the center point
    centers = get_kmeansplus_centers(M, data)
    # 2. Cluster according to the current center point
    clusters = create_clusters(centers, M, data)
    # 3. Calculate the new center point according to the clustering result of the previous step
    centers = calculate_centers(clusters, M, data)
    # 4. Clustering is conducted according to hierarchical Clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold).fit(centers)
    print(clustering.n_clusters_)
    centers = hierarchical_centers(clustering.children_, clustering.labels_, M, clustering.n_clusters_, data)
    # 5. Clustering was optimized according to Elkan k-means
    clusters = elkan_kmeans(clustering.n_clusters_, data, max_iterations, tol, centers)
    # Returns the final clustering label
    return get_cluster_labels(clusters, data)


if __name__ == '__main__':
    begin_time = time()
    X = txt2array("D:/python program/数据集/UCI/UCI数据集txt格式/txt/Ionosphere.txt", ",")
    n_samples, n_features = X.shape
    labels = HTK(X, 50, 0.01, 1000)
    end_time = time()
    run_time = end_time-begin_time
    print(u'Memory usage of the current process：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('The elapsed time of the loop program：', run_time)
    print(n_samples)
