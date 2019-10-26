import sys

import pandas as pd
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import cv2
from sklearn.metrics import davies_bouldin_score
import warnings
import matplotlib.patches as mpatches

plt.rcParams['figure.figsize'] = [12, 10]
plt.style.use('ggplot')

points_colors = {
    (237, 28, 36): ("red", 1),
    (255, 255, 255): ("white", 2)
}


def random_m(data, k: int):
    max_x, max_y = data[:, 0:1].max(), data[:, 1:2].max()
    min_x, min_y = data[:, 0:1].min(), data[:, 1:2].min()
    centroids = []
    for i in range(k):
        centroids.append([np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)])
    return np.array(centroids)


def random_partition(data, k: int):
    buckets = [[] for _ in range(9)]
    centroids = []
    for i in data:
        buckets[np.random.randint(0, k)].append([i[0], i[1]])
    for i in range(k):
        centroids.append(np.mean(buckets[i], axis=0))
    return np.array(centroids)


# def forgy_method(data, k: int):
#     centroids = data[np.random.choice(np.arange(len(data)), k, False)]
#     return centroids


def load_data_from_file(img_name: str):
    data_path = os.path.join(os.getcwd(), 'data')
    image_path = os.path.join(data_path, img_name)
    im = Image.open(image_path)
    pix = im.load()
    x_size, y_size = im.size
    print("Size: " + str(x_size) + ' ' + str(y_size))
    data = []
    for x in range(x_size):
        for y in range(y_size):
            if points_colors.get(pix[x, y])[0] is not 'white':
                data.append([x, y, points_colors.get(pix[x, y])[1]])
    data_set = pd.DataFrame(data,
                            columns={"x_coordinate", "y_coordinate", "color"})
    return data_set


def kmeans_test(X, n, n_clusters, init):
    warnings.filterwarnings('ignore')
    rep = 50
    index_score = []
    ns = []
    means = []
    stds = []
    for i in range(1, n):
        for j in range(rep):
            kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=i, n_init=1, tol=0).fit(X)
            predicted = kmeans.predict(X)
            index_score.append(davies_bouldin_score(X, predicted))
        score_mean = np.mean(index_score)
        score_std = np.std(index_score)
        ns.append(i)
        means.append(score_mean)
        stds.append(score_std / 5)
    #     result[i] = [score_mean, score_std]
    # print(result)
    print(stds[19])
    return ns, means, stds


def kmeans_test_2(X, max_iter, n_clusters):
    warnings.filterwarnings('ignore')
    rep = 50
    index_score = []
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=10).fit(X)
    predicted = kmeans.predict(X)
    #index_score.append(davies_bouldin_score(X, predicted))
    #score_mean = np.mean(index_score)
    score = davies_bouldin_score(X, predicted)
    print(score)
    return score #score_mean


def plot_chart(results):
    legend = []
    for i in results:
        plt.errorbar(results[i][0], results[i][1], results[i][2], markersize=1)
        # plt.plot(results[i][0], results[i][1])
        legend.append(i)
    plt.xlabel("number of iterations")
    plt.ylabel("Devies Bouldin mean score ")
    # plt.legend(["kmeans++", "forgy", "random_partition", "random"])
    plt.legend(legend)
    plt.show()


def plot_chart_for_k(results):
    legend = []
    for r in results:
        res = results[r]
        legend.append(r)
            #plt.errorbar(res[i][0], res[i][1], res[i][2], markersize=1)
        plt.plot(res[0], res[1])
    plt.xlabel("number of clusters")
    plt.ylabel("Devies Bouldin mean score ")
    plt.legend(legend)
    plt.show()


def score_to_n_ratio():
    files = ["firstDataSet_means.png", "secondDataSet_means.png"]
    for f in files:
        data_set = load_data_from_file(f)
        data = np.array(data_set.iloc[:, 0:2])
        max_iter = 25
        k = 9
        init = {"kmeans++": 'k-means++', "forgy": 'random', "random_partition": random_partition(data, k),
                "random": random_m(data, k)}

        results = {}
        for i in init:
            ns, values, stds = kmeans_test(data, max_iter, k, init=init[i])
            results[i] = [ns, values, stds]
        plot_chart(results)


def k_search():
    files = ["firstDataSet_means.png", "secondDataSet_means.png"]
    results = {}
    for f in files:
        data_set = load_data_from_file(f)
        data = np.array(data_set.iloc[:, 0:2])
        max_iter = 25
        k = 15
        k_list = []
        ratios = []

        for i in range(2, k):
            ratio = kmeans_test_2(data, max_iter, i)
            ratios.append(ratio)
            k_list.append(i)
        results[f] = [k_list, ratios]
    plot_chart_for_k(results)


if __name__ == '__main__':
    score_to_n_ratio()
    k_search()
