import argparse
import os
import random as rn
import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import umap
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import LocallyLinearEmbedding
from scipy.optimize import linear_sum_assignment as linear_assignment
from time import time
import helperFuncs as funcH

args_out = []

def eval_other_methods(x, y, args, names=None):
    global args_out
    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(x)
    y_pred_prob = gmm.predict_proba(x)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    args_out = funcH.print_and_add(args.dataset + " | GMM clustering on raw data", args_out)
    args_out = funcH.print_and_add('=' * 80, args_out)
    args_out = funcH.print_and_add(acc, args_out)
    args_out = funcH.print_and_add(nmi, args_out)
    args_out = funcH.print_and_add(ari, args_out)
    args_out = funcH.print_and_add('=' * 80, args_out)

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    args_out = funcH.print_and_add(args.dataset + " | K-Means clustering on raw data", args_out)
    args_out = funcH.print_and_add('=' * 80, args_out)
    args_out = funcH.print_and_add(acc, args_out)
    args_out = funcH.print_and_add(nmi, args_out)
    args_out = funcH.print_and_add(ari, args_out)
    args_out = funcH.print_and_add('=' * 80, args_out)

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    if args.manifold_learner == 'UMAP':
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(x)
    elif args.manifold_learner == 'LLE':
        from sklearn.manifold import LocallyLinearEmbedding
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(x)
    elif args.manifold_learner == 'tSNE':
        method = 'exact'
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(x)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(x)

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(hle)
    y_pred_prob = gmm.predict_proba(hle)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | GMM clustering on " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    save_dir = args.experiments_folder
    if args.visualize:
        n2d_plot(hle, y, 'UMAP', save_dir, args.dataset, args.n_clusters, names)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        n2d_plot(hle, y_pred_viz, 'UMAP-predicted', save_dir, args.dataset, args.n_clusters, names)

        return

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | K-Means " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

def n_learn_manifold(args, hidden_representation):
    global args_out
    args_out = funcH.print_and_add("Learning manifold(" + args.manifold_learner + ")", args_out)
    learn_time = time()
    if args.manifold_learner == 'UMAP':
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(hidden_representation)
    elif args.manifold_learner == 'LLE':
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(hidden_representation)
    elif args.manifold_learner == 'tSNE':
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(hidden_representation)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(hidden_representation)
    args_out = funcH.print_and_add("Time to learn manifold: " + str(time() - learn_time), args_out)
    return hle
def n_run_cluster(args, hle):
    global args_out
    args_out = funcH.print_and_add("Clustering("+args.cluster+")", args_out)
    cluster_time = time()
    if args.cluster == 'GMM':
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=args.n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif args.cluster == 'KM':
        km = KMeans(
            init='k-means++',
            n_clusters=args.n_clusters,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)
    elif args.cluster == 'SC':
        sc = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)
    args_out = funcH.print_and_add("Time to cluster: " + str(time() - cluster_time), args_out)
    return y_pred
def n_eval_result(definition_string, pngnameadd, args, hle, y, y_pred, label_names):
    global args_out
    y_pred = np.asarray(y_pred)
    # y_pred = y_pred.reshape(len(y_pred), )
    y = np.asarray(y)
    # y = y.reshape(len(y), )
    _confMat, kluster2Classes, kr_pdf, weightedPurity, cnmxh_perc = funcH.countPredictionsForConfusionMat(y, y_pred)
    sampleCount = np.sum(np.sum(_confMat))
    acc_doga = 100 * np.sum(np.diag(_confMat)) / sampleCount
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    args_out = funcH.print_and_add(definition_string + "-" + args.dataset + " | " + args.manifold_learner +
          " on autoencoded embedding with " + args.cluster + " - N2D", args_out)
    args_out = funcH.print_and_add("acc_doga(%6.3f),acc(%6.3f),nmi(%6.3f),ari(%6.3f)" % (acc_doga, acc, nmi, ari), args_out)
    if args.visualize:
        try:
            save_dir = args.experiments_folder
            n2d_plot(hle, y, 'n2d'+pngnameadd, save_dir, args.dataset, args.n_clusters, label_names)
            y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
            n2d_plot(hle, y_pred_viz, 'n2d-predicted'+pngnameadd, save_dir, args.dataset, args.n_clusters, label_names)
        except:
            args_out = funcH.print_and_add("couldnt visualize", args_out)
    return y_pred, acc, nmi, ari, acc_doga

def cluster_manifold_in_embedding(hl, y, args, label_names=None):
    global args_out
    args_out = funcH.print_and_add('=' * 80, args_out)
    y_pred_hl = n_run_cluster(args, hl)
    y_pred_hl, acc_hl, nmi_hl, ari_hl, acc_hl_dg = n_eval_result("if no manifold stuff", '-nm', args, hl, y, y_pred_hl, label_names)
    args_out = funcH.print_and_add('-' * 40, args_out)
    # find manifold on autoencoded embedding
    hle = n_learn_manifold(args, hl)
    # clustering on new manifold of autoencoded embedding
    y_pred_hle = n_run_cluster(args, hle)
    y_pred_hle, acc, nmi, ari, acc_dg = n_eval_result("hle", '-hle', args, hle, y, y_pred_hle, label_names)
    args_out = funcH.print_and_add('=' * 80, args_out)
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
    }
    return results_dict

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

def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    try:
        retval = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    except:
        retval = sum([w[ind[0][i], ind[1][i]] for i in range(len(ind[0]))]) * 1.0 / y_pred.size
    return retval

def n2d_plot(x, y, plot_id, save_dir, dataset_name, n_clusters, label_names=None):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if label_names is not None:
        viz_df['Label'] = viz_df['Label'].map(label_names)

    viz_df.to_csv(save_dir + '/' + dataset_name + '.csv')
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue='Label', legend='full', hue_order=sorted(viz_df['Label']),
                    palette=sns.color_palette("hls", n_colors=n_clusters),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=n_clusters + 1, handletextpad=0.01, )

    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(save_dir + '/' + dataset_name +
                '-' + plot_id + '.png', dpi=300)
    plt.clf()

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

def n_load_data(args):
    from .datasets import load_cifar10, load_mnist, load_mnist_test, load_usps, load_pendigits, load_fashion, load_har
    label_names = None
    if args.dataset == 'cifar10':
        x, y, label_names = load_cifar10()
    elif args.dataset == 'mnist':
        x, y = load_mnist()
    elif args.dataset == 'mnist-test':
        x, y = load_mnist_test()
    elif args.dataset == 'usps':
        x, y = load_usps()
    elif args.dataset == 'pendigits':
        x, y = load_pendigits()
    elif args.dataset == 'fashion':
        x, y, label_names = load_fashion()
    elif args.dataset == 'har':
        x, y, label_names = load_har()
    return x, y, label_names

def n_run_autoencode(x, args):
    global args_out
    # input_dict :
    # fit_verbose
    input_dict = argparse.ArgumentParser(description='func_autoencode', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_dict.add_argument("--ae_weights", default=None)
    input_dict.add_argument('--experiments_folder_base', default=funcH.getVariableByComputerName("n2d_experiments"))
    input_dict.add_argument('--weights_folder_base', default=funcH.getVariableByComputerName("n2d_experiments"))
    input_dict.add_argument('--weights_folder', default="weights_folder")
    input_dict.add_argument('--n_clusters', default=10, type=int)
    input_dict.add_argument('--dataset', default='mnist')
    input_dict.add_argument('--batch_size', default=256, type=int)
    input_dict.add_argument('--pretrain_epochs', default=100, type=int)
    input_dict.add_argument('--fit_verbose', default=True, type=bool)
    args = funcH._parse_args(input_dict, args, print_args=True)

    shape = [x.shape[-1], 500, 500, 2000, args.n_clusters]
    ae = _autoencoder(shape)
    hidden = ae.get_layer(name='encoder_%d' % (len(shape) - 2)).output
    encoder = Model(inputs=ae.input, outputs=hidden)

    weights_folder = os.path.join(args.weights_folder_base, args.weights_folder)
    pretrain_time = time()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        optimizer = 'adam'
        ae.compile(loss='mse', optimizer=optimizer)
        ae.fit(x, x, batch_size=args.batch_size, epochs=args.pretrain_epochs, verbose=1)
        pretrain_time = time() - pretrain_time
        funcH.createDirIfNotExist(weights_folder)
        weights_file = os.path.join(weights_folder, args.dataset + "-" + str(args.pretrain_epochs) + "-ae_weights.h5")
        ae.save_weights(weights_file)
        args_out = funcH.print_and_add("Time to train the ae: " + str(pretrain_time), args_out)
    else:
        weights_file = os.path.join(weights_folder, args.ae_weights)
        ae.load_weights(weights_file)

    funcH.createDirIfNotExist(args.experiments_folder_base)
    with open(os.path.join(args.experiments_folder_base, 'args_autoencode.txt'), 'w') as f:
        f.write("\n".join([str(k)+":"+str(args.__dict__[k]) for k in args.__dict__]))

    hl = encoder.predict(x)
    return hl

def init():
    global args_out
    args_out = []
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
        args_out = funcH.print_and_add("Using GPU", args_out)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1,
                                      )
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
    except BaseException:
        args_out = funcH.print_and_add("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.", args_out)
    np.set_printoptions(threshold=sys.maxsize)
    matplotlib.use('agg')

def get_args(argv):
    global args_out
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
    parser.add_argument('--eval_all', default=False, action='store_true')
    parser.add_argument('--manifold_learner', default='UMAP', type=str)
    parser.add_argument('--visualize', default=False, type=bool)
    args = funcH._parse_args(parser, argv, print_args=True)
    args_out = funcH.print_and_add('-' * 80)
    if args.ae_weights is not None:
        ae_cnt_str = '_lw_' + str(args.ae_weights).replace('-ae_weights.h5', '')
    else:
        ae_cnt_str = '_ae_' + str(args.dataset) + '-' + str(args.pretrain_epochs)
    args.exp_date_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M")).replace('-', '') #%S
    args.experiments_folder = os.path.join(args.experiments_folder_base, args.dataset, args.exp_date_str + ae_cnt_str + '_' + args.manifold_learner + '_' + args.cluster)
    funcH.createDirIfNotExist(args.experiments_folder)
    with open(os.path.join(args.experiments_folder, 'args_'+args.exp_date_str+'.txt'), 'w') as f:
        f.write("\n".join(argv))
    return args

def script():
    pretrain_epochs = 50
    manifold_learners_all = ["UMAP", "LLE", "tSNE", "isomap"]
    dataset_names_all = ["cifar10", "mnist", "usps", "pendigits", "fashion", "har"]
    for ds in dataset_names_all:
        for ml in manifold_learners_all:
            try:
                main(["--dataset", ds, "--gpu", "0",
                  "--pretrain_epochs", str(pretrain_epochs),
                  "--umap_dim", "20", "--umap_neighbors", "40",
                  "--manifold_learner", ml, "--umap_min_dist", "0.00"])
            except:
                global args_out
                args_out = funcH.print_and_add(ds + '_' + ml + " - problem", args_out)
                exp_date_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M")).replace('-', '')  # %S
                with open(os.path.join(funcH.getVariableByComputerName("n2d_experiments"), ds + '_' + ml + '_error_' + exp_date_str + '.txt'), 'w') as f:
                    f.write("\n".join(args_out))

def main(argv):
    init()
    args = get_args(argv)
    x, y, label_names = n_load_data(args)
    hl = n_run_autoencode(x, args)

    if args.eval_all:
        eval_other_methods(x, y, args, label_names)

    results_dict = cluster_manifold_in_embedding(hl, y, args, label_names)

    clusters_after_manifold = os.path.join(args.experiments_folder, args.dataset + '-clusters_after_manifold.txt')
    np.savetxt(clusters_after_manifold, results_dict["pred_after_manifold"], fmt='%i', delimiter=',')
    clusters_before_manifold = os.path.join(args.experiments_folder, args.dataset + '-clusters_before_manifold.txt')
    np.savetxt(clusters_before_manifold, results_dict["pred_before_manifold"], fmt='%i', delimiter=',')

    global args_out
    with open(os.path.join(args.experiments_folder, args.dataset + '-args_out.txt'), 'w') as f:
        f.write("\n".join(args_out))

    result_csv_file = os.path.join(args.experiments_folder_base, 'results.csv')
    result_row = [args.dataset, args.manifold_learner, args.cluster, "{:4.3f}".format(results_dict["acc_before_manifold_dg"]), "{:4.3f}".format(results_dict["acc_after_manifold_dg"])]
    print(result_row)
    if not os.path.isfile(result_csv_file):
        np.savetxt(result_csv_file, np.array(result_row).reshape(1, -1), fmt='%s', delimiter='*', newline=os.linesep,
               header='dataset * manifold * cluster * acc_bef * acc_aft', footer='', comments='', encoding=None)
    else:
        f = open(result_csv_file, 'a')
        np.savetxt(f, np.array(result_row).reshape(1, -1), fmt='s', delimiter='*', newline=os.linesep, header='',
                   footer='', comments='', encoding=None)
        f.close()


if __name__ == '__main__':
    # n2d.main("n2d.py", "mnist", "0", "--ae_weights", "mnist-1000-ae_weights.h5","--umap_dim", "10", "--umap_neighbors", "20", "--manifold_learner", "UMAP", "--save_dir", "mnist-n2d", "--umap_min_dist", "0.00")
    main(sys.argv)
