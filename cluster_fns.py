'''
Functions related to clustering
'''

import pandas as pd
import numpy as np
import lucem_illud_2020

import scipy
from sklearn_extra.cluster import KMedoids
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
import sklearn.mixture
from sklearn.model_selection import ParameterGrid

import gensim
import networkx as nx
from gensim.models import ldaseqmodel
from gensim.matutils import kullback_leibler

import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns


def make_vec_vectorizer(df):
    #initialize
    TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5,
                                                   max_features=1000,
                                                   min_df=3,
                                                   stop_words='english',
                                                   norm='l2')
    #train
    TFVects = TFVectorizer.fit_transform(df['text'])
    return TFVectorizer, TFVects


#### Functions for Flat Clustering ####

def make_train(numClusters, vects, algo):
    if algo == 'km':
        clf = sklearn.cluster.KMeans(n_clusters=numClusters,
                                     init='k-means++', n_jobs=-1)
    elif algo == 'dbscan':
        clf = sklearn.cluster.DBSCAN(metric='cosine', n_jobs=-1)
    elif algo == 'gauss':
        clf = sklearn.mixture.GaussianMixture(n_components=numClusters)
    clf.fit(vects)
    return clf


def cluster_labels(numClusters, km, df):
    df['kmeans_predictions'] = km.labels_
    for i in range(0, numClusters):
        print(f'Cluster: {i}')
        print(df[df.kmeans_predictions == i].head()['ep_title'])


def distinguish_features(Vectorizer, order_centroids, numClusters):
    # distinguishing features
    terms = Vectorizer.get_feature_names()
    print("Top terms per cluster:")
    for i in range(numClusters):
        print("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            print(f'{terms[ind]}')
        print('\n')
    return terms


def do_pca(TFVects):
    pca = sklearn.decomposition.PCA(n_components = 2).fit(TFVects.toarray())
    reduced_data = pca.transform(TFVects.toarray())
    components = pca.components_
    return reduced_data, components


def pca_and_plot(TFVects, TFVectorizer, clf, clf_name, cluster_num, labels):
    reduced_data, components = do_pca(TFVects)
    if clf_name == 'gauss':
        order_centroids = clf.means_.argsort()[:, ::-1]
    if clf_name == 'km':
        order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    keyword_ids = list(set(order_centroids[:,:10].flatten()))
    terms = distinguish_features(TFVectorizer, order_centroids, cluster_num)
    words = [terms[i] for i in keyword_ids]
    x = components[:,keyword_ids][0,:]
    y = components[:,keyword_ids][1,:]
    plot_clusters(reduced_data, cluster_num, labels, words, x, y)


def plot_dbscan(TFVects, Vectorizer, labels, words=False):
    reduced, components = do_pca(TFVects)
    clrs = sns.color_palette('husl', n_colors=20)
    color_dict = {}
    for i, val in enumerate(np.unique(labels)):
        color_dict[str(val)] = clrs[i]
    colors_p = [color_dict[str(l)] for l in labels]
    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=colors_p, alpha=0.3)
    if words:
        keyword_ids = np.where(TFVects.toarray()[:,1][labels] == 0)[0].tolist()
        terms = Vectorizer.get_feature_names()
        words = [terms[i] for i in keyword_ids]
        x = components[:,keyword_ids][0,:]
        y = components[:,keyword_ids][1,:]
        for i, word in enumerate(words):
            ax.annotate(word, (x[i],y[i]))

    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_clusters(reduced, cluster_num, labels, words, x, y):
    # color map for plots up to 6 clusters
    # for km labels=km.labels_

    colordict = {'0': 'red','1': 'orange', '2': 'green',
                 '3': 'blue', '4': 'purple', '5': 'yellow'}
    colors_p = [colordict[str(l)] for l in labels]
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    plt.scatter(reduced[:, 0], reduced[:, 1], color=colors_p, alpha=0.5)
    for i, word in enumerate(words):
        ax.annotate(word, (x[i],y[i]))
    plt.xticks(())
    plt.yticks(())
    plt.title(f'Predicted Clusters\n k = {cluster_num}')
    plt.show()


def plotSilhouette(n_clusters, X, clf_name, TFVects):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,5))
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    if clf_name == 'km':
        clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10, n_jobs=-1)
    elif clf_name == 'gauss':
        clusterer = sklearn.mixture.GaussianMixture(n_components=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels, metric='cosine')

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = matplotlib.cm.get_cmap("nipy_spectral")
        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    cmap = matplotlib.cm.get_cmap("nipy_spectral")
    colors = cmap(float(i) / n_clusters)


    pca = sklearn.decomposition.PCA(n_components=2).fit(TFVects.toarray())
    reduced_data = pca.transform(TFVects.toarray())
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    if clf_name == 'km':
        centers = clusterer.cluster_centers_
    elif clf_name == 'gauss':
        centers = clusterer.means_
    projected_centers = pca.transform(centers)
    # Draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.suptitle((f"Silhouette analysis for {clf_name} clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()
    print(f"For n_clusters = {n_clusters}, The average " + \
          "silhouette_score is : {:.3f}".format(silhouette_avg))


def plot_avg_sil(TFVects, clf_name):
    X = TFVects.toarray()
    k_range = range(2, 50)
    vals = []
    for k in k_range:
        if clf_name == 'km':
            clusterer = sklearn.cluster.KMeans(n_clusters=k, random_state=10, n_jobs=-1)
        if clf_name == 'gauss':
            clusterer = sklearn.mixture.GaussianMixture(n_components=k)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels, metric='cosine')
        vals.append(silhouette_avg)
    plt.plot(k_range,vals)
    plt.title('Avg Silhouette Score Over K Values (TAL)')


### Functions for Hierarchical Clustering ###

def make_coor_mat(TFVects):
    CoocMat = TFVects * TFVects.T
    CoocMat.setdiag(0)
    linkage_matrix = scipy.cluster.hierarchy.ward(CoocMat.toarray())
    return CoocMat, linkage_matrix

PARAMS_DICT = {
    'hier': {'p': [4], 'truncate_mode': ['level'], 'get_leaves': [True]},
    'flat': {'criterion': ['maxclust', 'distance'], 't': [2, 4, 6]}
    }

def make_dendos(CoocMat, linkage_matrix, cluster_type='hier', params=PARAMS_DICT):
    params_to_run = PARAMS_DICT[cluster_type]
    for param in ParameterGrid(params_to_run):
        print(cluster_type)
        print(param)
        param['Z'] = linkage_matrix
        if cluster_type == 'flat':
            d = scipy.cluster.hierarchy.fcluster(**param)
            print(sklearn.metrics.silhouette_score(CoocMat, d, metric='euclidean'))
        if cluster_type == 'hier':
            d = scipy.cluster.hierarchy.dendrogram(**param)

