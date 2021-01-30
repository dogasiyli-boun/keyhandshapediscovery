from helperFuncs import get_cluster_centroids, analyzeClusterDistribution, getElapsedTimeFormatted, removeLastLine
from pandas import DataFrame as pd_df
from numpy import array as np_array, asarray as np_asarray, unique as np_unique
from time import time
from datetime import datetime

from sklearn.cluster import KMeans, SpectralClustering #, OPTICS as ClusterOPT, cluster_optics_dbscan
from sklearn.mixture import GaussianMixture

from os import error as os_error

class Clusterer():
    #  TODO HDBSCAN should be added
    def __init__(self, cluster_model, n_clusters, spectral_affinity='nearest_neighbors', max_try_cnt=5):
        self.cluster_model = cluster_model
        self.n_clusters = n_clusters
        self.max_try_cnt = max_try_cnt
        self.trained_model = None
        if self.cluster_model in ['KMeans', 'KMEANS', 'km', 'kmeans', 'KM']:
            self.cluster_model = "KMeans"
        if self.cluster_model in ['GMM', 'GMM_Full', 'GMM_full']:
            self.cluster_model = "GMM_full"
        if self.cluster_model in ['GMMdiag', 'GMM_Diag', 'GMM_diag']:
            self.cluster_model = "GMM_diag"
        if self.cluster_model in ['Spectral', 'SC', 'SpectralClustering']:
            self.cluster_model = "Spectral"
            self.spectral_affinity = spectral_affinity  # 'nearest_neighbors', 'rbf'
        if self.cluster_model not in ["KMeans", "GMM_full", "GMM_diag", "Spectral"]:
            os_error(cluster_model + " is not applicable in this Class")

    def fit(self, X, post_analyze_distribution=False, verbose=1, random_state=0):
        df = pd_df(np_array(X))

        curTol = 0.0001 if self.cluster_model == 'KMeans' else 0.01
        max_iter = 300 if self.cluster_model == 'KMeans' else 200

        numOf_1_sample_bins = 1
        unique_clust_cnt = 1
        expCnt = 0
        while (unique_clust_cnt == 1 or numOf_1_sample_bins - expCnt > 0) and expCnt < self.max_try_cnt:
            t = time()
            if expCnt > 0:
                if numOf_1_sample_bins > 0:
                    print("running ", self.cluster_model, " for the ", str(expCnt), " time due to numOf_1_sample_bins(", str(numOf_1_sample_bins), ")")
                if unique_clust_cnt == 1:
                    print("running ", self.cluster_model, " for the ", str(expCnt), " time due to unique_clust_cnt==1")
            if verbose > 0:
                print('Clustering the featVec(', X.shape, ') with n_clusters(', str(self.n_clusters), ') and model = ',
                      self.cluster_model, ", curTol(", str(curTol), "), max_iter(", str(max_iter), "), at ",
                      datetime.now().strftime("%H:%M:%S"))
            self.kluster_centers = None
            self.predictedKlusters = None

            if self.cluster_model == 'KMeans':
                # default vals for kmeans --> max_iter=300, 1e-4
                self.trained_model = KMeans(n_clusters=self.n_clusters, n_init=5, tol=curTol, max_iter=max_iter, random_state=random_state).fit(df)
                self.predictedKlusters = self.trained_model.labels_.astype(float)
                self.kluster_centers = self.trained_model.cluster_centers_.astype(float)
            elif self.cluster_model == 'GMM_full':
                # default vals for gmm --> max_iter=100, 1e-3
                self.trained_model = GaussianMixture(n_components=self.n_clusters, covariance_type='full', tol=curTol, random_state=random_state, max_iter=max_iter).fit(df)
                _, log_resp = self.trained_model._e_step(X)
                self.predictedKlusters = log_resp.argmax(axis=1)
            elif self.cluster_model == 'GMM_diag':
                self.trained_model = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', tol=curTol, random_state=random_state, max_iter=max_iter).fit(df)
                _, log_resp = self.trained_model._e_step(X)
                self.predictedKlusters = log_resp.argmax(axis=1)
            elif self.cluster_model == 'Spectral':
                sc = SpectralClustering(n_clusters=self.n_clusters, affinity=self.spectral_affinity, random_state=random_state)
                self.trained_model = sc.fit(X)
                self.predictedKlusters = self.trained_model.labels_

            self.kluster_centroids = get_cluster_centroids(X, self.predictedKlusters, kluster_centers=self.kluster_centers, verbose=0)

            if post_analyze_distribution:
                numOf_1_sample_bins, histSortedInv = analyzeClusterDistribution(self.predictedKlusters, self.n_clusters, verbose=verbose)
                unique_clust_cnt = len(np_unique(self.predictedKlusters))
                curTol = curTol * 10
                max_iter = max_iter + 50
                expCnt = expCnt + 1
            else:
                expCnt = self.max_try_cnt

            elapsed = time() - t
            if verbose > 0:
                print('Clustering done in (', getElapsedTimeFormatted(elapsed), '), ended at ', datetime.now().strftime("%H:%M:%S"))
        removeLastLine()
        if verbose > 0:
            print('Clustering completed with (', np_unique(self.predictedKlusters).shape, ') clusters,  expCnt(', str(expCnt), ')')
        # elif 'OPTICS' in clusterModel:
        #     N = featVec.shape[0]
        #     min_cluster_size = int(np.ceil(N / (n_clusters * 4)))
        #     pars = clusterModel.split('_')  # 'OPTICS_hamming_dbscan', 'OPTICS_russellrao_xi'
        #     #  metricsAvail = np.sort(['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
        #     #                'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        #     #                'sokalsneath', 'sqeuclidean', 'yule',
        #     #                'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
        #     #  cluster_methods_avail = ['xi', 'dbscan']
        #     clust = ClusterOPT(min_samples=50, xi=.05, min_cluster_size=min_cluster_size, metric=pars[1], cluster_method=pars[2])
        #     clust.fit(featVec)
        #     predictedKlusters = cluster_optics_dbscan(reachability=clust.reachability_,
        #                                                core_distances=clust.core_distances_,
        #                                                ordering=clust.ordering_, eps=0.5)
        #     n1 = np.unique(predictedKlusters)
        #     print(clusterModel, ' found ', str(n1), ' uniq clusters')
        #     predictedKlusters = predictedKlusters + 1

        return self

    def fit_predict(self, X, post_analyze_distribution=False, verbose=1):
        self.fit(X, post_analyze_distribution=post_analyze_distribution, verbose=verbose)
        return np_asarray(self.predictedKlusters, dtype=int), self.kluster_centroids

    def predict(self, X, post_analyze_distribution, verbose=1):
        df = pd_df(X)
        if self.cluster_model == 'KMeans':
            # default vals for kmeans --> max_iter=300, 1e-4
            self.trained_model.predict(df)
            self.predictedKlusters = self.trained_model.labels_.astype(float)
            self.kluster_centers = self.trained_model.cluster_centers_.astype(float)
        elif self.cluster_model == 'GMM_full':
            # default vals for gmm --> max_iter=100, 1e-3
            _, log_resp = self.trained_model._e_step(df)
            self.predictedKlusters = log_resp.argmax(axis=1)
        elif self.cluster_model == 'GMM_diag':
            _, log_resp = self.trained_model._e_step(df)
            self.predictedKlusters = log_resp.argmax(axis=1)
        elif self.cluster_model == 'Spectral':
            self.predictedKlusters = self.trained_model.predict(X).labels_

        self.kluster_centroids = get_cluster_centroids(X, self.predictedKlusters, kluster_centers=self.kluster_centers, verbose=0)

        if post_analyze_distribution:
            numOf_1_sample_bins, histSortedInv = analyzeClusterDistribution(self.predictedKlusters, self.n_clusters,
                                                                            verbose=1)
            unique_clust_cnt = len(np_unique(self.predictedKlusters))
        return np_asarray(self.predictedKlusters, dtype=int), self.kluster_centroids