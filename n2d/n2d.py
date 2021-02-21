import argparse
import os
import random as rn
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
import helperFuncs as funcH
import projRelatedHelperFuncs as prHF
from manifoldLearner import ManifoldLearner
from clusteringWrapper import Clusterer

debug_string_out = []

# args.experiment_names_and_folders - adopted
def n2d_plot(x, y, clusters_count, plot_id, file_name_plot_fig_full=None, file_name_plot_csv_full=None, label_names=None):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if label_names is not None:
        viz_df['Label'] = viz_df['Label'].map(label_names)

    if file_name_plot_csv_full is not None:
        viz_df.to_csv(file_name_plot_csv_full.replace(".csv", "_" + str(plot_id) + "_.csv"))
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label']),
                    palette=sns.color_palette("hls", n_colors=clusters_count),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=clusters_count + 1, handletextpad=0.01, )

    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    if file_name_plot_fig_full is not None:
        plt.savefig(file_name_plot_fig_full.replace("<plot_id>", str(plot_id)), dpi=300)
        plt.clf()
    plt.show()

# args.experiment_names_and_folders - adopted
def n_learn_manifold(hidden_representation, embedding_dim, manifold_learner="UMAP", manifold_out_file_name=None, optional_params=None):
    min_dist = funcH.get_attribute(optional_params, "umap_min_dist", default_type=float, default_val=0.0)
    distance_metric = funcH.get_attribute(optional_params, "umap_metric", default_type=str, default_val='euclidean')
    num_of_neighbours = funcH.get_attribute(optional_params, "umap_neighbors", default_type=int, default_val=20)
    mfl = ManifoldLearner(manifold_dimension=embedding_dim, manifold_learner=manifold_learner,
                          min_dist=min_dist, distance_metric=distance_metric, num_of_neighbours=num_of_neighbours)
    manifold_feats, dbg_str = mfl.learn_manifold(X=hidden_representation, manifold_out_file_name=manifold_out_file_name)
    global debug_string_out
    debug_string_out = debug_string_out + dbg_str
    return manifold_feats

def n_run_cluster(hle, n_clusters, cluster_func_name='GMM', experiment_names_and_folders=None, file_name_add=""):
    global debug_string_out
    debug_string_out = funcH.print_and_add("Clustering(" + cluster_func_name + ")" + str(datetime.now()), debug_string_out)
    t = funcH.Timer()
    if experiment_names_and_folders is not None:
        file_name_cluster_obj = experiment_names_and_folders["file_name_cluster_obj"].replace("<bef_aft>", file_name_add)
        if os.path.isfile(file_name_cluster_obj):
            cluster_obj = funcH.load_dict_fr_file(file_name_cluster_obj)
        else:
            cluster_obj = Clusterer(cluster_model=cluster_func_name, n_clusters=n_clusters, max_try_cnt=1).fit(hle, post_analyze_distribution=True, verbose=1)
            try:
                funcH.print_fancy("Trying to dump " + file_name_cluster_obj, style="Bold", textColor="Black", backColor='White', end='\n')
                funcH.dump_dict_to_file(file_name_cluster_obj, cluster_obj)
            except Exception as e:
                print(str(e))
                funcH.print_fancy("Couldn't dump " + file_name_cluster_obj, style="Bold", textColor="Red", end='\n')
    else:
        cluster_obj = Clusterer(cluster_model=cluster_func_name, n_clusters=n_clusters, max_try_cnt=1).fit(hle, post_analyze_distribution=True, verbose=1)

    y_pred, kluster_centroids = cluster_obj.predict(hle, post_analyze_distribution=True, verbose=1)
    debug_string_out = funcH.print_and_add("Time to cluster: " + t.get_elapsed_time(), debug_string_out)
    return y_pred, kluster_centroids
def n_get_acc(y, y_pred, centroid_info_pdf=None):
    _confMat, kluster2Classes, kr_pdf, weightedPurity, cnmxh_perc = funcH.countPredictionsForConfusionMat(y, y_pred, centroid_info_pdf=centroid_info_pdf)
    sampleCount = np.sum(np.sum(_confMat))
    _acc = 100 * np.sum(np.diag(_confMat)) / sampleCount
    return _acc
def n_eval_result(hle, y, y_pred, label_names, cluster_func_name, clusters_count, dataset_name, definition_string, pngnameadd, experiment_names_and_folders, optional_params=None, visualize=False):
    global debug_string_out
    y_pred = np.asarray(y_pred)
    # y_pred = y_pred.reshape(len(y_pred), )
    y = np.asarray(y)
    # y = y.reshape(len(y), )
    manifold_learner = funcH.get_attribute(optional_params, "manifold_learner", default_type=str, default_val='UMAP')

    kluster_centroids = funcH.get_cluster_centroids(hle, y_pred, kluster_centers=None, verbose=0)
    acc_doga = n_get_acc(y, y_pred, centroid_info_pdf=kluster_centroids)
    acc_doga_wo_kluster_centroids = n_get_acc(y, y_pred, centroid_info_pdf=None)
    debug_string_out = funcH.print_and_add("acc_doga:" + "{:6.4f}".format(acc_doga) + ", acc_doga_wo_kluster_centroids:" + "{:6.4f}".format(acc_doga_wo_kluster_centroids), debug_string_out)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    debug_string_out = funcH.print_and_add(definition_string + "-" + dataset_name + " | " + manifold_learner +
          " on autoencoded embedding with " + cluster_func_name + " - N2D", debug_string_out)
    debug_string_out = funcH.print_and_add("acc_doga(%6.3f),acc(%6.3f),nmi(%6.3f),ari(%6.3f)" % (acc_doga, acc, nmi, ari), debug_string_out)
    if visualize:
        try:
            n2d_plot(hle, y, clusters_count=clusters_count, plot_id='n2d'+pngnameadd,
                     file_name_plot_fig_full=experiment_names_and_folders["file_name_plot_fig_full"],
                     file_name_plot_csv_full=experiment_names_and_folders["file_name_plot_csv_full"],
                     label_names=label_names)
            y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
            n2d_plot(hle, y_pred_viz, clusters_count=clusters_count, plot_id='n2d-predicted'+pngnameadd,
                     file_name_plot_fig_full=experiment_names_and_folders["file_name_plot_fig_full"],
                     file_name_plot_csv_full=experiment_names_and_folders["file_name_plot_csv_full"],
                     label_names=label_names)
        except:
            debug_string_out = funcH.print_and_add("could not visualize", debug_string_out)
    return y_pred, acc, nmi, ari, acc_doga

# args.experiment_names_and_folders - no need to adopt
def cluster_manifold_in_embedding(hl, y, cluster_func_name, clusters_count, dataset_name, experiment_names_and_folders, label_names=None, optional_params=None):
    global debug_string_out
    debug_string_out = funcH.print_and_add('=' * 80, debug_string_out)

    umap_dim = funcH.get_attribute(optional_params, "umap_dim", default_type=int, default_val=clusters_count)
    manifold_learner = funcH.get_attribute(optional_params, "manifold_learner", default_type=str, default_val="UMAP")

    funcH.add_attribute(optional_params, "umap_min_dist", funcH.get_attribute(optional_params, "umap_min_dist", default_type=float, default_val=0.0))
    funcH.add_attribute(optional_params, "umap_metric", funcH.get_attribute(optional_params, "umap_metric", default_type=str, default_val='euclidean'))
    funcH.add_attribute(optional_params, "umap_neighbors", funcH.get_attribute(optional_params, "umap_neighbors", default_type=int, default_val=20))
    y_pred_hl, kluster_centroids_before = n_run_cluster(hl, n_clusters=clusters_count, cluster_func_name=cluster_func_name, experiment_names_and_folders=experiment_names_and_folders, file_name_add="-nm")
    y_pred_hl, acc_hl, nmi_hl, ari_hl, acc_hl_dg = n_eval_result(hl, y, y_pred_hl, label_names,
                                                                 cluster_func_name, clusters_count,
                                                                 dataset_name, definition_string="if no manifold stuff",
                                                                 pngnameadd='-nm',
                                                                 experiment_names_and_folders=experiment_names_and_folders,
                                                                 optional_params=None, visualize=False)
    try:
        funcH.print_fancy("Trying to save " + experiment_names_and_folders["file_name_data_before_manifold"], style="Bold", textColor="Black", backColor='White', end='\n')
        np.savez(experiment_names_and_folders["file_name_data_before_manifold"], featVec=hl, labels=y, preds=y_pred_hl, label_names=label_names, acc=acc_hl_dg)
    except Exception as e:
        print(str(e))
        funcH.print_fancy("Couldn't save " + experiment_names_and_folders["file_name_data_before_manifold"], style="Bold", textColor="Red", end='\n')
    debug_string_out = funcH.print_and_add('-' * 40, debug_string_out)

    file_name_silhouette_results = experiment_names_and_folders["file_name_silhouette_results"].replace("<bef_aft>", "before")
    if os.path.isfile(file_name_silhouette_results):
        print("loading silhouette_values from ", file_name_silhouette_results)
        silhouette_values_hl = np.load(file_name_silhouette_results, allow_pickle=True)
    else:
        _, silhouette_values_hl = funcH.calc_silhouette_params(hl, y_pred_hl)
        print("saving silhouette_values to ", file_name_silhouette_results)
        try:
            funcH.print_fancy("Trying to save " + file_name_silhouette_results, style="Bold", textColor="Black", backColor='White', end='\n')
            np.save(file_name_silhouette_results, silhouette_values_hl, allow_pickle=True)
        except Exception as e:
            print(str(e))
            funcH.print_fancy("Couldn't save " + file_name_silhouette_results, style="Bold", textColor="Red", end='\n')

    # find manifold on autoencoded embedding
    hle = n_learn_manifold(hl, embedding_dim=umap_dim, manifold_learner=manifold_learner,
                           manifold_out_file_name=experiment_names_and_folders["file_name_umap_data_full"],
                           optional_params=optional_params)
    # clustering on new manifold of autoencoded embedding
    y_pred_hle, kluster_centroids_after = n_run_cluster(hle, n_clusters=clusters_count, cluster_func_name=cluster_func_name, experiment_names_and_folders=experiment_names_and_folders, file_name_add="-hle")
    y_pred_hle, acc, nmi, ari, acc_dg = \
        n_eval_result(hle, y, y_pred_hle, label_names, cluster_func_name, clusters_count,
                      dataset_name, definition_string="after manifold stuff",
                      pngnameadd='-hle', experiment_names_and_folders=experiment_names_and_folders,
                      optional_params=None, visualize=False)
    saveToFileName = experiment_names_and_folders["file_name_data_after_manifold"]
    try:
        funcH.print_fancy("Trying to save " + saveToFileName, style="Bold", textColor="Black",
                          backColor='White', end='\n')
        np.savez(saveToFileName, featVec=hle, labels=y, preds=y_pred_hle, label_names=label_names, acc=acc_dg)
    except Exception as e:
        print(str(e))
        funcH.print_fancy("Couldn't save " + saveToFileName, style="Bold", textColor="Red", end='\n')

    debug_string_out = funcH.print_and_add('=' * 80, debug_string_out)

    file_name_silhouette_results = experiment_names_and_folders["file_name_silhouette_results"].replace("<bef_aft>", "after")
    if os.path.isfile(file_name_silhouette_results):
        print("loading silhouette_values from ", file_name_silhouette_results)
        silhouette_values_hle = np.load(file_name_silhouette_results, allow_pickle=True)
    else:
        _, silhouette_values_hle = funcH.calc_silhouette_params(hle, y_pred_hle)
        print("saving silhouette_values to ", file_name_silhouette_results)
        try:
            funcH.print_fancy("Trying to save " + file_name_silhouette_results, style="Bold", textColor="Black",
                              backColor='White', end='\n')
            np.save(file_name_silhouette_results, silhouette_values_hle, allow_pickle=True)
        except Exception as e:
            print(str(e))
            funcH.print_fancy("Couldn't save " + file_name_silhouette_results, style="Bold", textColor="Red", end='\n')



    results_dict = {
        "acc_before_manifold": acc_hl,
        "acc_before_manifold_dg": acc_hl_dg,
        "acc_after_manifold_dg": acc_dg,
        "acc_after_manifold": acc,
        "nmi_before_manifold": nmi_hl,
        "nmi_after_manifold": nmi,
        "ari_before_manifold": ari_hl,
        "ari_after_manifold": ari,
        "pred_before_manifold": y_pred_hl,
        "pred_after_manifold": y_pred_hle,
        "silhouette_values_before": silhouette_values_hl,
        "silhouette_values_after": silhouette_values_hle,
        "kluster_centroids_before": kluster_centroids_before,
        "kluster_centroids_after": kluster_centroids_after,
    }
    return results_dict

# args.experiment_names_and_folders - no need to adopt
def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w

# args.experiment_names_and_folders - no need to adopt
def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    try:
        retval = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    except:
        retval = sum([w[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size
    return retval

# args.experiment_names_and_folders - no need to adopt
def _autoencoder(dims, act='relu'):
    n_stacks = len(dims) - 1
    x = Input(shape=(dims[0],), name='input')
    h = x
    for i in range(n_stacks - 1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)
    h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)
    for i in range(n_stacks - 1, 0, -1):
        h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)
    h = Dense(dims[0], name='decoder_0')(h)

    return Model(inputs=x, outputs=h)
# args.experiment_names_and_folders - adopted
def n_run_autoencode(x, args):
    global debug_string_out
    # input_dict :
    # fit_verbose
    input_dict = argparse.ArgumentParser(description='func_autoencode', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_dict.add_argument('--experiments_folder_base', default=funcH.getVariableByComputerName("n2d_experiments"))
    input_dict.add_argument('--n_clusters', default=10, type=int)
    input_dict.add_argument('--dataset', default='mnist')
    input_dict.add_argument('--batch_size', default=256, type=int)
    input_dict.add_argument('--pretrain_epochs', default=100, type=int)
    input_dict.add_argument('--fit_verbose', default=True, type=bool)
    args2 = funcH._parse_args(input_dict, args, print_args=True)

    shape = [x.shape[-1], 500, 500, 2000, args2.n_clusters]
    ae = _autoencoder(shape)
    hidden = ae.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=ae.input, outputs=hidden)

    print("checking if ", args.experiment_names_and_folders["file_name_ae_weights_full"], " exist.")
    weights_file = args.experiment_names_and_folders["file_name_ae_weights_full"]
    load_file_skip_learning = os.path.isfile(weights_file)
    t = funcH.Timer()

    # Pretrain autoencoders before clustering
    if load_file_skip_learning:
        debug_string_out = funcH.print_and_add("Load weigths from(" + weights_file + ")", debug_string_out)
        ae.load_weights(weights_file)
    else:
        optimizer = 'adam'
        ae.compile(loss='mse', optimizer=optimizer)
        ae.fit(x, x, batch_size=args2.batch_size, epochs=args2.pretrain_epochs, verbose=1)
        t.end()
        ae.save_weights(weights_file)
        debug_string_out = funcH.print_and_add("Time to train the ae: " + t.get_elapsed_time(), debug_string_out)

    with open(args.experiment_names_and_folders["file_name_ae_params_text_full"], 'w') as f:
        f.write("\n".join([str(k)+":"+str(args2.__dict__[k]) for k in args2.__dict__]))

    hl = encoder.predict(x)
    return hl

# args.experiment_names_and_folders - no need to adopt
def n_load_data(dataset_name):
    from .datasets import load_cifar10, load_mnist, load_mnist_test, load_usps, load_pendigits, load_fashion, load_har
    label_names = None
    if dataset_name == 'cifar10':
        x, y, label_names = load_cifar10()
    elif dataset_name == 'mnist':
        x, y = load_mnist()
    elif dataset_name == 'mnist-test':
        x, y = load_mnist_test()
    elif dataset_name == 'usps':
        x, y = load_usps()
    elif dataset_name == 'pendigits':
        x, y = load_pendigits()
    elif dataset_name == 'fashion':
        x, y, label_names = load_fashion()
    elif dataset_name == 'har':
        x, y, label_names = load_har()
    elif "_" in dataset_name:
        dataIdent, pca_dim, nos = str(dataset_name).split('_')
        x, labels_all, labels_sui, labels_map = prHF.combine_pca_hospisign_data(dataIdent=dataIdent, pca_dim=int(pca_dim),
                                                                                       nos=int(nos), verbose=2)
        y = np.asarray(labels_all, dtype=int).squeeze()
        label_names = np.asarray(labels_map["khsName"])
    return x, y, label_names

# args.experiment_names_and_folders - no need to adopt
def init():
    global debug_string_out
    debug_string_out.clear()
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    sns.set_context("paper", font_scale=1.3)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # str(sys.argv[2])
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

    rn.seed(0)
    try:
        tf.set_random_seed(seed=0)
    except:
        tf.random.set_seed(seed=0)
    np.random.seed(0)

    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        debug_string_out = funcH.print_and_add("Using GPU", debug_string_out)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1,
                                      )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
    except BaseException:
        debug_string_out = funcH.print_and_add("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.", debug_string_out)
    np.set_printoptions(threshold=sys.maxsize)
    matplotlib.use('agg')

# args.experiment_names_and_folders - adopted
def get_args(argv):
    global debug_string_out
    parser = argparse.ArgumentParser(
        description='(Not Too) Deep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist', )
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--experiments_folder_base', default=funcH.getVariableByComputerName("n2d_experiments"))
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=52162)
    parser.add_argument('--gpu', default=0, )
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=1000, type=int)
    parser.add_argument('--umap_dim', default=2, type=int)
    parser.add_argument('--umap_neighbors', default=10, type=int)
    parser.add_argument('--umap_min_dist', default="0.00", type=str)
    parser.add_argument('--umap_metric', default='euclidean', type=str)
    parser.add_argument('--cluster', default='GMM', type=str)
    parser.add_argument('--manifold_learner', default='UMAP', type=str)
    parser.add_argument('--visualize', default=False, type=bool)
    args = funcH._parse_args(parser, argv, print_args=True)
    debug_string_out = funcH.print_and_add('-' * 80)

    experiment_names_and_folders = {
        "exp_date_str": str(datetime.now().strftime("%Y%m%d_")).replace('-', ''),  # %M%S,
        "exp_base_str": "_".join([args.dataset, "c" + str(args.cluster)+ str(args.n_clusters), "e" + str(args.pretrain_epochs)]),
        "folder_umap_data": os.path.join(args.experiments_folder_base, "exported_manifolds"),
        "folder_ae_weights": os.path.join(args.experiments_folder_base, "weights"),
    }
    experiment_names_and_folders["exp_extended"] = experiment_names_and_folders["exp_base_str"] + "_" + "_".join([args.manifold_learner + "ud" + str(args.umap_dim), "un" + str(args.umap_neighbors)])
    experiment_names_and_folders["folder_experiment"] = os.path.join(args.experiments_folder_base, args.dataset,
                                           experiment_names_and_folders["exp_date_str"] + experiment_names_and_folders["exp_extended"])
    experiment_names_and_folders["file_name_ae_weights_base"] = "aew_" + "_".join([args.dataset, "c" + str(args.n_clusters), "e" + str(args.pretrain_epochs)])
    experiment_names_and_folders["file_name_ae_weights_full"] = os.path.join(experiment_names_and_folders["folder_ae_weights"], experiment_names_and_folders["file_name_ae_weights_base"] + '.npy')
    experiment_names_and_folders["file_name_umap_data_base"] = "ulp" + experiment_names_and_folders["exp_extended"]
    experiment_names_and_folders["file_name_umap_data_full"] = os.path.join(experiment_names_and_folders["folder_umap_data"], experiment_names_and_folders["file_name_umap_data_base"] + '.npy')
    experiment_names_and_folders["file_name_arguments_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'args_' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.txt')
    experiment_names_and_folders["file_name_ae_params_text_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'args_autoencode_' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.txt')
    experiment_names_and_folders["file_name_plot_fig_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'plot_' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '_<plot_id>.png')
    experiment_names_and_folders["file_name_plot_csv_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'csv_' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.csv')
    experiment_names_and_folders["file_name_clusters_after_manifold_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'clusters_after_manifold-' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.txt')
    experiment_names_and_folders["file_name_clusters_before_manifold_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'clusters_before_manifold-' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.txt')
    experiment_names_and_folders["file_name_debug_string_out_full"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'debug_string_out-' + experiment_names_and_folders["exp_extended"] + "_" + experiment_names_and_folders["exp_date_str"] + '.txt')
    experiment_names_and_folders["file_name_result_csv_file_full"] = os.path.join(args.experiments_folder_base, 'results.csv')
    experiment_names_and_folders["file_name_data_before_manifold"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'data_' + experiment_names_and_folders["exp_extended"] + '_before.npz')
    experiment_names_and_folders["file_name_data_after_manifold"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'data_' + experiment_names_and_folders["exp_extended"] + '_after.npz')
    experiment_names_and_folders["file_name_cluster_obj"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'cluster_obj_' + experiment_names_and_folders["exp_extended"] + '_<bef_aft>.dictionary')
    experiment_names_and_folders["file_name_silhouette_results"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'silhouette_results_' + experiment_names_and_folders["exp_extended"] + '_<bef_aft>.npy')
    experiment_names_and_folders["file_name_results"] = os.path.join(experiment_names_and_folders["folder_experiment"], 'results_' + experiment_names_and_folders["exp_extended"] + '.dictionary')

    args.experiment_names_and_folders = experiment_names_and_folders

    # 4 folders folder_{experiment, umap_data, ae_weights}
    funcH.createDirIfNotExist(experiment_names_and_folders["folder_experiment"])
    funcH.createDirIfNotExist(experiment_names_and_folders["folder_umap_data"])
    funcH.createDirIfNotExist(experiment_names_and_folders["folder_ae_weights"])

    with open(experiment_names_and_folders["file_name_arguments_full"], 'w') as f:
        f.write("\n".join(argv))
    return args

def script():
    global debug_string_out
    pretrain_epochs = [10, 50]
    manifold_learners_all = ["UMAP"]
    dataset_names_all = ["cifar10", "mnist", "pendigits", "fashion"]  # , "usps", "har"
    cluster_func = "HDBSCAN"
    for ds in dataset_names_all:
        for ml in manifold_learners_all:
            for ae_epoc in pretrain_epochs:
                for clust_cnt in [20]: #  umap_dim = 20, n_clusters_ae = 20, umap_neighbors = 40
                    try:
                        debug_string_out = []
                        main(["--dataset", ds, "--gpu", "0",
                              "--pretrain_epochs", str(ae_epoc),
                              "--n_clusters", str(clust_cnt), '--cluster', str(cluster_func),
                              "--umap_dim", str(clust_cnt), "--umap_neighbors", str(2*clust_cnt),
                              "--manifold_learner", ml, "--umap_min_dist", "0.00"])
                    except Exception as e:
                        debug_string_out = funcH.print_and_add(ds + '_' + ml + " - problem \n" + str(e), debug_string_out)
                        exp_date_str = str(datetime.now().strftime("%Y%m%d_%H%M")).replace('-', '')  # %S
                        with open(os.path.join(funcH.getVariableByComputerName("n2d_experiments"), ds + '_' + ml + '_error_' + exp_date_str + '.txt'), 'w') as f:
                            f.write("\n".join(debug_string_out))

def script_hgsk():
    global debug_string_out
    pretrain_epochs = [10]
    ml = "UMAP"
    ds = "hgsk_256_41"
    for cluster in ['KM', 'GMM']:
        for ae_epoc in pretrain_epochs:
            for clust_cnt in [512, 1024]: #  umap_dim = 20, n_clusters_ae = 20, umap_neighbors = 40
                for umap_neighbors in [20, 30, 40]:
                    try:
                        debug_string_out.clear()
                        main(["--dataset", ds, "--gpu", "0",
                              "--pretrain_epochs", str(ae_epoc),
                              "--n_clusters", str(clust_cnt), "--cluster", cluster,
                              "--umap_dim", str(clust_cnt), "--umap_neighbors", str(umap_neighbors),
                              "--manifold_learner", ml, "--umap_min_dist", "0.00"])
                    except Exception as e:
                        debug_string_out = funcH.print_and_add(ds + '_' + ml + " - problem", debug_string_out)
                        debug_string_out = funcH.print_and_add(str(e), debug_string_out)
                        exp_date_str = str(datetime.now().strftime("%Y%m%d_%H%M")).replace('-', '')  # %S
                        with open(os.path.join(funcH.getVariableByComputerName("n2d_experiments"), ds + '_' + ml + '_error_' + exp_date_str + '.txt'), 'w') as f:
                            f.write("\n".join(debug_string_out))

# args.experiment_names_and_folders - adopted
def append_to_results(args, results_dict):
    result_csv_file = args.experiment_names_and_folders["file_name_result_csv_file_full"]
    result_row = [args.dataset, str(args.n_clusters), str(args.umap_dim), str(args.umap_neighbors), str(args.pretrain_epochs), args.manifold_learner, args.cluster, "{:4.3f}".format(results_dict["acc_before_manifold_dg"]), "{:4.3f}".format(results_dict["acc_after_manifold_dg"])]
    print(result_row)
    if not os.path.isfile(result_csv_file):
        np.savetxt(result_csv_file, np.array(result_row).reshape(1, -1), fmt='%s', delimiter='*', newline=os.linesep,
               header='dataset * aeClustCnt * umapClustCnt * umapNeighbour * aeEpoch * manifold * cluster * acc_bef * acc_aft', footer='', comments='', encoding=None)
    else:
        f = open(result_csv_file, 'a')
        np.savetxt(f, np.array(result_row).reshape(1, -1), fmt='%s', delimiter='*', newline=os.linesep, header='',
                   footer='', comments='', encoding=None)
        f.close()

def main(argv):
    init()
    global debug_string_out
    debug_string_out.clear()
    args = get_args(argv)
    debug_string_out_file = args.experiment_names_and_folders["file_name_debug_string_out_full"]
    if os.path.isfile(debug_string_out_file):
        print("skipping experiment already done (" + debug_string_out_file + ")")
        return
    x, y, label_names = n_load_data(args.dataset)
    hl = n_run_autoencode(x, args)

    if os.path.isfile(args.experiment_names_and_folders["file_name_results"]):
        results_dict = funcH.load_dict_fr_file(args.experiment_names_and_folders["file_name_results"], "Results")
    else:
        results_dict = cluster_manifold_in_embedding(hl, y, cluster_func_name=args.cluster,
                                                     clusters_count=args.n_clusters,
                                                     dataset_name=args.dataset,
                                                     experiment_names_and_folders=args.experiment_names_and_folders,
                                                     label_names=label_names, optional_params=args)
        funcH.dump_dict_to_file(args.experiment_names_and_folders["file_name_results"], results_dict, "Results")

    np.savetxt(args.experiment_names_and_folders["file_name_clusters_after_manifold_full"], results_dict["pred_after_manifold"], fmt='%i', delimiter=',')
    np.savetxt(args.experiment_names_and_folders["file_name_clusters_before_manifold_full"], results_dict["pred_before_manifold"], fmt='%i', delimiter=',')

    conf_plot_save_to = args.experiment_names_and_folders["file_name_plot_fig_full"].replace("<plot_id>", "conf_before")
    funcH.analyze_silhouette_values(results_dict["silhouette_values_before"], results_dict["pred_before_manifold"], y, centroid_info_pdf=results_dict["kluster_centroids_before"], label_names=label_names, conf_plot_save_to=conf_plot_save_to)
    conf_plot_save_to = args.experiment_names_and_folders["file_name_plot_fig_full"].replace("<plot_id>", "conf_after")
    funcH.analyze_silhouette_values(results_dict["silhouette_values_after"], results_dict["pred_after_manifold"], y, centroid_info_pdf=results_dict["kluster_centroids_after"], label_names=label_names, conf_plot_save_to=conf_plot_save_to)

    append_to_results(args, results_dict)

    with open(debug_string_out_file, 'w') as f:
        f.write("\n".join(debug_string_out))


if __name__ == '__main__':
    # n2d.main("n2d.py", "mnist", "0", "--ae_weights", "mnist-1000-ae_weights.h5","--umap_dim", "10", "--umap_neighbors", "20", "--manifold_learner", "UMAP", "--save_dir", "mnist-n2d", "--umap_min_dist", "0.00")
    main(sys.argv)
