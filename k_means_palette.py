import os
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, silhouette_samples


# plt.rcParams['figure.figsize'] = [12, 10]
# plt.style.use('ggplot')


def load_data_from_file():
    data_path = os.path.join(os.getcwd(), 'data')
    image_path = os.path.join(data_path, "data_pic.jpg")
    im = Image.open(image_path)
    pix = im.load()
    x_size, y_size = im.size
    print("Size: " + str(x_size) + ' ' + str(y_size))
    data = []
    for x in range(x_size):
        for y in range(y_size):
            # if points_colors.get(pix[x, y])[0] is not 'white':
            data.append(np.array(pix[x, y]))
    data_set = pd.DataFrame(data,
                            columns={"R", "G", "B"})
    return data_set


def k_maedoids_centers(X, k, init="k-means++"):
    # kmeans = KMeans(n_clusters=k, max_iter=1, n_init=1, init=init).fit(X)
    kmean = KMeans(n_clusters=k).fit(X)
    centers = []
    index_score = []
    score_mean = -2
    score_std = -2
    index = -1
    indexes = []
    for i in range(10):
        for center in kmean.cluster_centers_:
            new_center = X[0]
            dist = np.linalg.norm(center - new_center)
            for i in range(1, X.__len__()):
                tmp_dist = np.linalg.norm(center - X[i])
                if tmp_dist < dist:
                    index = i
                    dist = tmp_dist
                    new_center = X[i]
            centers.append(new_center)
            indexes.append(index)
        predicted = kmean.predict(X)
        index_score.append(davies_bouldin_score(X, predicted))
        score_mean = np.mean(index_score)
        score_std = np.std(index_score)
    kmean.cluster_centers_ = np.array(centers)
    return kmean, score_mean, score_std, np.array(indexes)


def k_medoids(X, k):
    init = k_maedoids_centers(X, k)
    # init = np.array(init)
    # kmeans = KMeans(n_clusters=k, max_iter=1, n_init=1, init=init).fit(X)
    # for i in range(50):
    #     centers = kmeans.cluster_centers_
    #     init = k_maedoids_centers(X, k, init=centers)
    #     init = np.array(init)
    #     kmeans = KMeans(n_clusters=k, max_iter=1, n_init=1, init=init).fit(X)
    # predicted = kmeans.predict(X)
    # index = davies_bouldin_score(X, predicted)
    # print(index)


def plot_comparison(cent_num, score_mean_list, score_std_list):
    plt.errorbar(cent_num, score_mean_list, score_std_list, markersize=1)
    plt.xlabel("k clasters")
    plt.ylabel("Devies Bouldin mean score ")
    plt.show()


def test_kmedoids(X):
    centers_list = []
    kmeans = []
    score_mean_list = []
    score_std_list = []
    cent_num = []
    for k in range(3, 15):
        kmean, score_mean, score_std, indexes = k_maedoids_centers(X, k)
        cent_num.append(k)
        kmeans.append(kmean)
        score_mean_list.append(score_mean)
        score_std_list.append(score_std)
    plot_comparison(cent_num, score_mean_list, score_std_list)
    return kmeans[score_mean_list.index(min(score_mean_list))]


def color_pallette(kmeans):
    plt.title("Paleta kolorów")
    plt.imshow(np.array(kmeans.cluster_centers_))
    plt.show()


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    data_set = load_data_from_file()
    X = np.array(data_set.iloc[:, 0:3])
    X_unique = np.unique(X, axis=0)
    p = np.array([1, 2, 3, 4])
    print(p[:-1])
    # test_kmedoids(X_unique)
    # print(X_unique)
    kmeans, score_mean, score_std, indexes = k_maedoids_centers(X_unique, 9)
    color_pallette(kmeans)

    # pca = PCA(n_components=2)
    # pca.fit(X_unique)
    # X_w = pca.transform(X_unique)
    # # print(X_unique)
    # # print(kmeans.cluster_centers_[kmeans.labels_])
    # #
    # # plt.figure(figsize=(14, 14))
    # # plt.scatter(X_w[:, 0], X_w[:, 1], c=X_unique / 255.0, s=100)
    # # plt.scatter(X_w[:, 0], X_w[:, 1], edgecolors=kmeans.cluster_centers_[kmeans.labels_] / 255.0, facecolors='none',
    # #             s=100)
    # # plt.show()
    #
    # silhouette = silhouette_samples(X_unique, kmeans.labels_)
    # # print(silhouette)
    # plt.figure(figsize=(14, 14))
    # s = silhouette + np.full(len(silhouette), 1, dtype=np.float64)
    # ind = np.arange(0,len(silhouette))
    # # print(s)
    # # print(s[0] * [2, 2, 2])
    # c = []
    # conv = [127, 127, 127]
    # conv = np.array(conv, dtype=np.float64)
    # for si in s:
    #     point = []
    #     for ci in conv:
    #         point.append(ci * si)
    #     c.append(point)
    # print(c)
    # c = np.array(c)
    # a = np.full(len(silhouette), 0.5)
    # plt.scatter(X_w[:, 0], X_w[:, 1], c=silhouette[ind]/255.0, s=100)
    # plt.scatter(X_w[indexes][:, 0], X_w[indexes][:, 1], c=X_unique[indexes] / 255.0, s=100)
    # plt.scatter(X_w[indexes][:, 0], X_w[indexes][:, 1], facecolors='none', edgecolors='black', s=100)
    # plt.title("Wykres Silhouette")
    # plt.show()
