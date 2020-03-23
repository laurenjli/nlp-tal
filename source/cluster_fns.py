'''
Functions related to clustering and topic modeling
'''

import pandas as pd
import numpy as np
import lucem_illud_2020

import scipy
from sklearn_extra.cluster import KMedoids
import sklearn
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import CountVectorizer
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


def dropMissing(wordLst, vocab):
    '''
    Helper function to drop words if not in specified list
    '''
    return [w for w in wordLst if w in vocab]


def make_vec_vectorizer(df):
    '''
    Converts a collection of raw documents to a matrix of TF-IDF features
    '''
    #initialize
    TFVectorizer = sklearn.feature_extraction. \
                           text.TfidfVectorizer(max_df=0.5, max_features=1000,
                                                min_df=2, stop_words='english',
                                                norm='l2')
    #train
    TFVects = TFVectorizer.fit_transform(df['text'])
    return TFVectorizer, TFVects


#### Functions for Flat Clustering ####

def make_train(numClusters, vects, algo):
    '''
    Initializes classifier and fit classifier with training data
    '''
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
    '''
    Prints labels associated with clusters
    '''
    df['kmeans_predictions'] = km.labels_
    for i in range(0, numClusters):
        print(f'Cluster: {i}')
        print(df[df.kmeans_predictions == i].head()['ep_title'])


def distinguish_features(Vectorizer, order_centroids, numClusters):
    '''
    Finds the most relevent terms for each cluster
    '''
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
    '''
    Completes principal component analysis to lower dimensionality of data
    '''
    pca = sklearn.decomposition.PCA(n_components = 2).fit(TFVects.toarray())
    reduced_data = pca.transform(TFVects.toarray())
    components = pca.components_
    return reduced_data, components


def pca_and_plot(TFVects, TFVectorizer, clf, clf_name, cluster_num, labels,
                 print_words=True):
    '''
    Completes PCA on text data, finds relevant features and then plots a
    2D rendering of clusters with these features annotated on plot
    '''
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
    plot_clusters(reduced_data, cluster_num, labels, words, x, y, print_words)


def plot_dbscan(TFVects, Vectorizer, labels, words=False):
    '''
    Completes PCA and plots clusters for DBSCAN clustering
    '''
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


def plot_clusters(reduced, cluster_num, labels, words, x, y, print_words):
    '''
    Plots clusters with feature labels
    '''
    clrs = sns.color_palette('husl', n_colors=20)
    color_dict = {}
    for i, val in enumerate(np.unique(labels)):
        color_dict[str(val)] = clrs[i]
    colors_p = [color_dict[str(l)] for l in labels]
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    plt.scatter(reduced[:, 0], reduced[:, 1], color=colors_p, alpha=0.5)
    if print_words:
        for i, word in enumerate(words):
            ax.annotate(word, (x[i],y[i]))
    plt.xticks(())
    plt.yticks(())
    plt.title(f'Predicted Clusters\n k = {cluster_num}')
    plt.show()


def plotSilhouette(n_clusters, X, clf_name, TFVects):
    '''
    Finds silhouette score for each cluster by fitting the classifier
    with testing data, plots silhouette score for each cluster,
    and finds and prints average silhouette score for all clusters
    '''
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,5))
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    if clf_name == 'km':
        clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                           random_state=10, n_jobs=-1)
    elif clf_name == 'gauss':
        clusterer = sklearn.mixture.GaussianMixture(n_components=n_clusters)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels,
                                                      metric='cosine')
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
    # PCA to get 2D rendering
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
    plt.suptitle((f"Silhouette analysis for {clf_name} clustering on " + \
                  f"sample data with n_clusters = {n_clusters}"),
                 fontsize=14, fontweight='bold')
    plt.show()
    print(f"For n_clusters = {n_clusters}, The average " + \
          "silhouette_score is : {:.3f}".format(silhouette_avg))


def plot_avg_sil(TFVects, clf_name):
    '''
    Plots average silhouette score for clusters over a range of k values
    (2 through 50 clusters)
    '''
    X = TFVects.toarray()
    k_range = range(2, 50)
    vals = []
    for k in k_range:
        if clf_name == 'km':
            clusterer = sklearn.cluster.KMeans(n_clusters=k, random_state=10,
                                               n_jobs=-1)
        if clf_name == 'gauss':
            clusterer = sklearn.mixture.GaussianMixture(n_components=k)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels,
                                                          metric='cosine')
        vals.append(silhouette_avg)
    plt.plot(k_range, vals)
    plt.title('Avg Silhouette Score Over K Values (TAL)')


### Functions for Hierarchical Clustering ###

def make_coor_mat(TFVects):
    '''
    Makes cooccurance matrix and completes hierarchical clustering
    using Wald's method
    '''
    CoocMat = TFVects * TFVects.T
    CoocMat.setdiag(0)
    linkage_matrix = scipy.cluster.hierarchy.ward(CoocMat.toarray())
    return CoocMat, linkage_matrix


PARAMS_DICT = {
    'hier': {'p': [4], 'truncate_mode': ['level'], 'get_leaves': [True]},
    'flat': {'criterion': ['maxclust', 'distance'], 't': [2, 4, 6]}
    }


def make_dendos(CoocMat, linkage_matrix, cluster_type='hier',
                params=PARAMS_DICT):
    '''
    Plots dendograms for hierarchical and flat clustering, using a
    variety of parameters
    '''
    params_to_run = PARAMS_DICT[cluster_type]
    for param in ParameterGrid(params_to_run):
        print(cluster_type)
        print(param)
        param['Z'] = linkage_matrix
        if cluster_type == 'flat':
            d = scipy.cluster.hierarchy.fcluster(**param)
            print(sklearn.metrics.silhouette_score(CoocMat, d,
                                                   metric='euclidean'))
        if cluster_type == 'hier':
            d = scipy.cluster.hierarchy.dendrogram(**param)


### TOPIC MODELING FUNCTIONS ###


def plot_stacked_heat(tal_lda, ldaDFVis, ldaDFVisNames, t, heatmap=None):
    '''
    Plots a stacked bar chart for each documents breakdown into topics as
    defined by tal_lda model
    '''
    N = t
    ind = np.arange(N)
    K = tal_lda.num_topics  # N documents, K topics
    ind = np.arange(N)  # the x-axis locations for the novels
    width = 0.5  # the width of the bars
    plots = []
    height_cumulative = np.zeros(N)

    for k in range(K):
        color = plt.cm.coolwarm(k/K, 1)
        if k == 0:
            p = plt.bar(ind, ldaDFVis[:, k], width, color=color)
        else:
            p = plt.bar(ind, ldaDFVis[:, k], width,
                        bottom=height_cumulative, color=color)
        height_cumulative += ldaDFVis[:, k]
        plots.append(p)

    # proportions sum to 1, so the height of the stacked bars is 1
    plt.ylim((0, 1))
    plt.ylabel('Topics')
    plt.title('Topics in Press Releases')
    plt.xticks(ind+width/2, ldaDFVisNames,
               rotation='vertical')
    plt.yticks(np.arange(0, 1, 10))
    topic_labels = ['Topic #{}'.format(k) for k in range(K)]
    plt.legend([p[0] for p in plots], topic_labels, loc='center left',
               frameon=True,  bbox_to_anchor = (1, .5))
    plt.show()
    if heatmap:
        plt.pcolor(ldaDFVis, norm=None, cmap='Blues')
        plt.yticks(np.arange(ldaDFVis.shape[0])+0.5, ldaDFVisNames);
        plt.xticks(np.arange(ldaDFVis.shape[1])+0.5, topic_labels);

        # flip the y-axis so the texts are in the order we anticipate
        plt.gca().invert_yaxis()
        # rotate the ticks on the x-axis
        plt.xticks(rotation=90)
        # add a legend
        plt.colorbar(cmap='Blues')
        plt.tight_layout()  # fixes margins
        plt.show()
    word_ranks = make_topic_df(tal_lda)
    print(word_ranks)


def make_lda_model(df, tokens_col, num_tops=10, fil_ex=True):
    '''
    Creates a Latent Dirichlet allocation model with specified params
    '''
    dictionary, bow_corpus = make_dictionary(df, tokens_col)
    if fil_ex:
        dictionary.filter_extremes(no_below=10, keep_n=100000)
    tal_lda = gensim.models.LdaMulticore(bow_corpus, num_topics=num_tops,
                                         id2word=dictionary, passes=5,
                                         workers=4)
    return dictionary, tal_lda, corpus


def make_dictionary(df, tokens_col):
    '''
    Converts tokens cells to gensim dictionary object and
    makes corpus a bag of words 
    '''
    dictionary = gensim.corpora.Dictionary(df[tokens_col])
    corpus = [dictionary.doc2bow(text) for text in df[tokens_col]]
    return dictionary, corpus


def make_top_probs_df(df, id_col, dictionary, lda, tokens_col):
    '''
    Finds topic probabilities for new documents based of lda model
    '''
    ldaDF = pd.DataFrame({'name' : df[id_col],
                      'topics' : [lda[dictionary.doc2bow(l)] for \
                                  l in df[tokens_col]]})
    #Dict to temporally hold the probabilities
    topicsProbDict = {i : [0] * len(ldaDF) for i in range(lda.num_topics)}

    #Load them into the dict
    for index, topicTuples in enumerate(ldaDF['topics']):
        for topicNum, prob in topicTuples:
            topicsProbDict[topicNum][index] = prob

    #Update the DataFrame
    for topicNum in range(lda.num_topics):
        ldaDF[f'topic_{topicNum}'] = topicsProbDict[topicNum]
    return ldaDF

        
def make_topic_df(tal_lda):
    '''
    Creates a dataframe of the top words for each topic
    '''
    topicsDict = {}
    for topicNum in range(tal_lda.num_topics):
        topicWords = [w for w, p in tal_lda.show_topic(topicNum)]
        topicsDict[f'Topic_{topicNum}'] = topicWords
    wordRanksDF = pd.DataFrame(topicsDict)
    return wordRanksDF


def topic_distribution(df, col, model):
    '''
    Finds the breakdown of topics for each row in df from previously
    created lda model
    
    (a percentage for each topic, with these percentages summing to
    1 for each row)
    '''
    corpora = df[col].apply(lambda x: gensim.matutils. \
                                             Sparse2Corpus(vect.transform([x]),
                                                           documents_columns=False))
    output = corpora.apply(lambda x: list(model[x]))
    return output.to_dict()


def new_model_fn(df, col, num_tops=10, min_docs=0.2, max_docs=0.8):
    '''
    Complete lda topic modeling using multiple cores to speed up process
    '''
    vect = CountVectorizer(min_df=min_docs, max_df=max_docs,
                           stop_words='english')
    # Fit and transform
    X = vect.fit_transform(df[col])
    # Convert sparse matrix to gensim corpus.
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    # Mapping from word IDs to words
    id_map = dict((v, k) for k, v in vect.vocabulary_.items())
    # Use the gensim.models.ldamodel.LdaModel constructor to estimate
    # LDA model parameters on the corpus, and save to the variable `ldamodel`
    ldamodel = gensim.models.ldamulticore. \
                             LdaMulticore(corpus, num_topics=num_tops,
                                          id2word=id_map, passes=5, workers=4,
                                          random_state=34)
    output = ldamodel.print_topics(num_tops, 3)
    print(min_docs, max_docs)
    print(output)
    return ldamodel, output
