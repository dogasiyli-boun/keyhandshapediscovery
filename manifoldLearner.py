import helperFuncs as funcH

from time import time
from datetime import datetime
from os.path import isfile
from numpy import load as np_load, save as np_save

from umap import UMAP
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap

class ManifoldLearner():
    def __init__(self, manifold_dimension, manifold_learner="UMAP", min_dist=0.0, distance_metric="euclidean", num_of_neighbours=10):
        self.debug_string_out = []
        self.manifold_dimension = manifold_dimension
        self.manifold_learner = manifold_learner
        self.min_dist = min_dist
        self.dist_metric = distance_metric
        self.num_of_neighbours = num_of_neighbours

    def print_and_remember(self, print_str):
        self.debug_string_out = funcH.print_and_add(print_str, self.debug_string_out)

    def learn_manifold(self, X, manifold_out_file_name=None):
        self.debug_string_out.clear()
        self.print_and_remember("Learning manifold(" + self.manifold_learner + ")" + str(datetime.now()))
        learn_time = time()

        if manifold_out_file_name is not None and isfile(manifold_out_file_name):  # check the learned manifold existance
            manifold_feats = np_load(manifold_out_file_name, allow_pickle=True)
            self.print_and_remember("Manifold loaded(" + manifold_out_file_name + ")")
        elif self.manifold_learner == 'UMAP':
            manifold_feats = UMAP(
                random_state=0,
                metric=self.dist_metric,
                n_components=self.manifold_dimension,
                n_neighbors=self.num_of_neighbours,
                min_dist=float(self.min_dist)).fit_transform(X)
        elif self.manifold_learner == 'LLE':
            manifold_feats = LocallyLinearEmbedding(
                n_components=self.manifold_dimension,
                n_neighbors=self.num_of_neighbours).fit_transform(X)
        elif self.manifold_learner == 'tSNE':
            manifold_feats = TSNE(
                n_components=self.manifold_dimension,
                random_state=0,
                verbose=0).fit_transform(X)
        elif self.manifold_learner == 'isomap':
            manifold_feats = Isomap(n_components=self.manifold_dimension,
                                    n_neighbors=self.num_of_neighbours).fit_transform(X)
        self.print_and_remember("Time to learn manifold: " + str(funcH.getElapsedTimeFormatted(time() - learn_time)))
        if manifold_out_file_name is not None:
            np_save(manifold_out_file_name, manifold_feats, allow_pickle=True)
            self.print_and_remember("Manifold saved(" + manifold_out_file_name + ")")
        return manifold_feats, self.debug_string_out
