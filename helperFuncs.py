import socket
import os
import sys
import numpy as np
from math import isnan as isNaN
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans, SpectralClustering #, OPTICS as ClusterOPT, cluster_optics_dbscan
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame as pd_df
import pandas as pd
import scipy.io
import time
import datetime

from collections import Counter

def npy_to_matlab(folderOfNPYFiles, matFileName):
    #  downloaded from https://github.com/ruitome/npy_to_matlab
    # npy_to_matlab('/home/doga/Desktop/hgsk', 'm_Hgsk')
    files = (os.listdir(folderOfNPYFiles))
    npyFiles = []
    matStructure = {}

    for f in files:
        extension = os.path.splitext(f)[1]
        if extension == '.npy':
            npyFiles.append(f)

    if not npyFiles:
        print("Error: There are no .npy files in %s folder".format(folderOfNPYFiles))
        sys.exit(0)

    for f in npyFiles:
        currentFile = os.path.join(folderOfNPYFiles, f)
        variable = os.path.splitext(f)[0]

        # MATLAB only loads variables that start with normal characters
        variable = variable.lstrip('0123456789.-_ ')

        try:
            values = np.load(currentFile, allow_pickle=True)
        except IOError:
            print("Error: can\'t find file or read data", currentFile)

        else:
            matStructure[variable] = values

    matFileName = os.path.join(folderOfNPYFiles, matFileName + '.mat')

    if matStructure:
        scipy.io.savemat(matFileName, matStructure)

def install_package_str(package_name):
    return "!{sys.executable} - m pip install " + package_name

def removeLastLine():
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")

def getElapsedTimeFormatted(elapsed_miliseconds):
    hours, rem = divmod(elapsed_miliseconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        retStr = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    elif minutes > 0:
        retStr = "{:0>2}:{:05.2f}".format(int(minutes), seconds)
    else:
        retStr = "{:05.2f}".format(seconds)
    return retStr

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

def getFolderList(dir2Search, startString="", endString="", sortList=False):
    folderList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isdir(os.path.join(dir2Search, f))]
    if sortList:
        folderList = np.sort(folderList)
    return folderList

def filterList(string_list, includeList, excludeList):
    filter_func = lambda s: any(x in s for x in includeList) and not any(x in s for x in excludeList)
    matching_lines = [line for line in string_list if filter_func(line)]
    return matching_lines

def numOfFilesInFolder(dir2Search, startswith="", endswith=""):
    numOfFiles = len(getFileList(dir2Search, startString=startswith, endString=endswith))
    return numOfFiles

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def append_to_vstack(stack, new_arr, dtype=int):
    if (stack is None or len(stack)==0):
        stack = new_arr
    else:
        stack = np.asarray(np.vstack((stack, new_arr)), dtype=dtype)
    return stack

def printMatRes(X, defStr):
    print(defStr, '\n', X)
    print('samples-rows : ', X.shape[0], 'feats-cols : ', X.shape[1])
    mean0_X = np.mean(X, axis=0)
    mean1_X = np.mean(X, axis=1)
    print('mean over samples,axis=0', mean0_X, ', cnt=', len(mean0_X))
    print('correct --> mean over features,axis=1', mean1_X, ', cnt=', len(mean1_X))

def setPandasDisplayOpts():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_rows', None)

def createLabelsForConfMat(confMat):
    r, c = confMat.shape
    rowLabels = []
    colLabels = []
    for ri in range(r):
        for ci in range(c):
            numOfSamples = int(confMat[ri, ci])
            for i in range(numOfSamples):
                rowLabels.append(ri)
                colLabels.append(ci)
    return rowLabels, colLabels

def removeConfMatUnnecessaryRows(_confMat):
    _confMat = _confMat[~np.all(_confMat == 0, axis=1)]
    return _confMat

def getVariableByComputerName(variableName):
    curCompName = socket.gethostname()
    if variableName=='base_dir':
        if curCompName == 'doga-MSISSD':
            base_dir = '/media/doga/SSD258/DataPath'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            base_dir = '/media/dg/SSD_Data/DataPath'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            base_dir = '/home/doga/DataFolder'  # for laptop
        else:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = base_dir
    if variableName=='data_dir':
        if curCompName == 'doga-MSISSD':
            data_dir = '/media/doga/SSD258/DataPath/bdData'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            data_dir = '/media/dg/SSD_Data/DataPath/bdData'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            data_dir = '/home/doga/DataFolder/bdData'  # for laptop
        else:
            data_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = data_dir
    if variableName=='results_dir':
        if curCompName == 'doga-MSISSD':
            results_dir = '/media/doga/SSD258/DataPath/bdResults'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            results_dir = '/media/dg/SSD_Data/DataPath/bdResults'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            results_dir = '/home/doga/DataFolder/bdResults'  # for laptop
        else:
            results_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = results_dir
    return retVal

def createDirIfNotExist(dir2create):
    if not os.path.isdir(dir2create):
        os.makedirs(dir2create)

def normalize(x, axis=None):
    x = x / np.linalg.norm(x, axis=axis)
    return x

def discretizeW(W, printAssignments=False):
    rows2ColAssignments = np.argmax(W, axis=1) + 1
    if printAssignments:
        print("rows2ColAssignments Assignments: ", rows2ColAssignments)
    W_discrete = np.zeros(W.shape, dtype=int)
    for i in range(W.shape[0]):
        W_discrete[i, rows2ColAssignments[i] - 1] = 1
    return W_discrete, rows2ColAssignments

def calcCleanConfMat(labels, predictions):
    xtoclear = confusion_matrix(labels, predictions)
    x_cleaned = xtoclear[np.any(xtoclear, axis=1), :]
    return x_cleaned

def confusionFromKluster(labVec, predictedKlusters):
    _confMat = []
    #rows are true labels, cols are predictedLabels

    _confMat = calcCleanConfMat(labVec, predictedKlusters)
    #_confMat = removeConfMatUnnecessaryRows(_confMat)

    inputConfMat_const = tf.constant(_confMat, dtype="float")
    c_C, r_K = _confMat.shape
    symb_W = tf.Variable(tf.truncated_normal(shape=[r_K, c_C], stddev=0.1, ), dtype="float")
    W_softMax = tf.nn.softmax(symb_W, 1)
    regularizerCoeff = tf.constant(10.0, dtype="float")
    symbOutConfMat = tf.einsum('ck,kx->cx', inputConfMat_const, W_softMax)  # eXpectedClassCount
    symbOutCost = -tf.trace(symbOutConfMat)
    regularizar = tf.reduce_sum(tf.square(tf.reduce_sum(W_softMax, axis=0))) + tf.reduce_sum(
    tf.square(tf.reduce_sum(W_softMax, axis=1)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(symbOutCost + regularizerCoeff * regularizar)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for eph in range(1000):
            sess.run(optimizer)
        W = sess.run((W_softMax))
    w_discrete, kluster2Classes = discretizeW(W)
    _confMat = _confMat @ w_discrete
    return _confMat, kluster2Classes

def accFromKlusterLabels(labVec, predictedKlusters, removeZeroLabels=False):
    labVec = np.asarray(labVec,dtype=int)
    predictedKlusters = np.asarray(predictedKlusters,dtype=int)
    if removeZeroLabels:
        predictedKlusters = predictedKlusters[np.where(labVec)]
        labVec = labVec[np.where(labVec)]

    _confMat, kluster2Classes = confusionFromKluster(labVec, predictedKlusters)
    classCntPrecision = 1.0
    acc = np.trace(_confMat)/np.einsum('ij->', _confMat)
    return acc, classCntPrecision

def getAccFromConf(labels, predictions):
    inputConfMat = calcCleanConfMat(labels, predictions)
    c_C, r_K = inputConfMat.shape
    expectedClassCount = c_C
    # print('expectedClassCount(c_C-', expectedClassCount, ') -- Rows(', r_K, '-r_K) are Klusters', ', Cols(c_C-', c_C ,') are Classes in this case')
    inputConfMat_const = tf.constant(inputConfMat, dtype="float")

    symb_W = tf.Variable(tf.truncated_normal(shape=[r_K, expectedClassCount], stddev=0.1, ), dtype="float")
    # print('symb_W(', r_K, ',', expectedClassCount,') will give me the mapping of klusters to classes')
    W_softMax = tf.nn.softmax(symb_W, 1)
    regularizerCoeff = tf.constant(10.0, dtype="float")
    symbOutConfMat = tf.einsum('ck,kx->cx', inputConfMat_const, W_softMax)  # eXpectedClassCount
    symbOutCost = -tf.trace(symbOutConfMat)
    regularizar = tf.reduce_sum(tf.square(tf.reduce_sum(W_softMax, axis=0))) + tf.reduce_sum(
        tf.square(tf.reduce_sum(W_softMax, axis=1)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.5).minimize(symbOutCost + regularizerCoeff * regularizar)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for eph in range(1000):
            sess.run(optimizer)
        W = sess.run((W_softMax))
    W_discrete, Kluster2Classes = discretizeW(W)
    confMat = inputConfMat @ W_discrete;
    # print("Confusion Mat:\n",confMat)
    # for n in range(1,expectedClassCount+1):
    #   Kn = np.where(Kluster2Classes == n)[0]+1
    #   print(("K{} "*Kn.size).format(*Kn), "<--", " C", n)
    # rowLabels, colLabels = createLabelsForConfMat(inputConfMat)

    acc = np.sum(np.diag(confMat)) / np.sum(np.sum(confMat))
    # acc = np.trace(confMat) / np.einsum('ij->', confMat)

    # nmiAr  = nmi(colLabels,rowLabels,average_method='arithmetic')
    # nmiGeo = nmi(colLabels,rowLabels,average_method='geometric')
    return acc
    # print('confAcc(', acc ,'),nmiAr(', nmiAr,')nmiGeo(',nmiGeo ,')')

def get_nmi_only(l, p, average_method='geometric'):
    nmi_res = nmi(l, p, average_method=average_method)
    return nmi_res

def get_NMI_Acc(non_zero_labels, non_zero_predictions, average_method='geometric'):
    nmi_cur = get_nmi_only(non_zero_labels, non_zero_predictions, average_method=average_method)
    acc_cur = getAccFromConf(non_zero_labels, non_zero_predictions)
    return nmi_cur, acc_cur

def get_nmi_deepCluster(featVec, labVec, n_clusters, clusterModel='KMeans', normMode='', applyPca=True):
    predictedKlusters = clusterData(featVec, n_clusters,
                                    applyPca=applyPca, normMode=normMode,
                                    clusterModel=clusterModel)
    nmi_score = get_nmi_only(labVec, predictedKlusters, average_method='geometric')
    labVec_nonzero, predictedKlusters_nonzero = getNonZeroLabels(labVec, predictedKlusters)
    nmi_score_nonzero = get_nmi_only(labVec_nonzero, predictedKlusters_nonzero, average_method='geometric')
    return nmi_score, predictedKlusters, nmi_score_nonzero

def applyMatTransform(featVec, applyPca=True, whiten=True, normMode='', verbose=0):
    exp_var_rat = []
    if applyPca:
        #pca = PCA(whiten=whiten, svd_solver='full')
        pca = PCA(whiten=False, svd_solver='auto')
        featVec = pca.fit_transform(featVec)
        exp_var_rat = np.cumsum(pca.explained_variance_ratio_)
        if verbose > 0:
            print('Max of featsPCA = ', np.amax(featVec), ', Min of featsPCA = ', np.amin(featVec))

    if normMode == '':
        pass # do nothing
    elif normMode == 'nm':
        featVec = normalize(featVec, axis=0)  # divide per column max
    elif normMode == 'nl':
        featVec = normalize(featVec, axis=None)  # divide to a scalar = max of matrix
    else:
        os.error("normMode must be defined as one of the following = ['','nm','nl']. normMode(" + normMode + ")")

    if verbose > 0 and normMode != '':
        print('Max of normedFeats = ', np.amax(featVec), ', Min of normedFeats = ', np.amin(featVec))

    return featVec, exp_var_rat
    # X = np.array([[3, 5, 7, 9, 11], [4, 6, 15, 228, 245], [28, 19, 225, 149, 81], [18, 9, 125, 49, 2181], [8, 9, 25, 149, 81], [8, 9, 25, 49, 81], [8, 19, 25, 49, 81]])
    # print('input array : \n', X)
    # print('samples-rows : ', X.shape[0], ' / feats-cols : ', X.shape[1])
    #
    # X_nT_pF = applyMatTransform(X, applyNormalization=True, applyPca=False)
    # printMatRes(X_nT_pF,'X_nT_pF')
    #
    # X_nF_pT = applyMatTransform(X, applyNormalization=False, applyPca=True)
    # printMatRes(X_nF_pT,'X_nF_pT')
    #
    # X_nT_pT = applyMatTransform(X, applyNormalization=True, applyPca=True)
    # printMatRes(X_nT_pT,'X_nT_pT')

def getNonZeroLabels(labVec, predictedKlusters, detailedLabels=None):
    labVec_nz = np.asarray(labVec[np.where(labVec)], dtype=int)
    predictedKlusters_nz = np.asarray(predictedKlusters[np.where(labVec)], dtype=int)
    if detailedLabels is not None:
        detailedLabels_nz = detailedLabels[np.where(labVec), :].squeeze()
    else:
        detailedLabels_nz = detailedLabels
    return labVec_nz, predictedKlusters_nz, detailedLabels_nz

def clusterData(featVec, n_clusters, normMode='', applyPca=True, clusterModel='KMeans'):
    featVec, exp_var_rat = applyMatTransform(np.array(featVec), applyPca=applyPca, normMode=normMode)
    df = pd_df(featVec)

    curTol = 0.0001 if clusterModel == 'KMeans' else 0.01
    max_iter = 300 if clusterModel == 'KMeans' else 200

    t = time.time()
    numOf_1_sample_bins = 1
    expCnt = 0
    while numOf_1_sample_bins > 0 and expCnt < 5:
        if expCnt > 0:
            print("running ", clusterModel, " for the ", str(expCnt), " time due to numOf_1_sample_bins(",
                  str(numOf_1_sample_bins), ")")
        print('Clustering the featVec(', featVec.shape, ') with n_clusters(', str(n_clusters), ') and model = ',
              clusterModel, ", curTol(", str(curTol), "), max_iter(", str(max_iter), "), at ",
              datetime.datetime.now().strftime("%H:%M:%S"))
        if clusterModel == 'KMeans':
                #default vals for kmeans --> max_iter=300, 1e-4
                kmeans_result = KMeans(n_clusters=n_clusters, n_init=5, tol=curTol, max_iter=max_iter).fit(df)
                predictedKlusters = kmeans_result.labels_.astype(float)
        elif clusterModel == 'GMM_full':
            # default vals for gmm --> max_iter=100, 1e-3
            predictedKlusters = GaussianMixture(n_components=n_clusters, covariance_type='full', tol=curTol, max_iter=max_iter).fit_predict(df)
        elif clusterModel == 'GMM_diag':
            predictedKlusters = GaussianMixture(n_components=n_clusters, covariance_type='diag', tol=curTol, max_iter=max_iter).fit_predict(df)
        elif clusterModel == 'Spectral':
            sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', random_state=0)
            sc_clustering = sc.fit(featVec)
            predictedKlusters = sc_clustering.labels_
        numOf_1_sample_bins, histSortedInv = analyzeClusterDistribution(predictedKlusters, n_clusters, verbose=0)
        curTol = curTol * 10
        max_iter = max_iter + 50
        expCnt = expCnt + 1
        elapsed = time.time() - t
        print('Clustering done in (', getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))
    removeLastLine()
    print('Clustering completed with (', np.unique(predictedKlusters).shape, ') clusters,  expCnt(', str(expCnt), ')')
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

    return np.asarray(predictedKlusters, dtype=int)

def backtrack(D, max_x, max_y):
    #https://github.com/gulzi/DTWpy/blob/master/dtwpy.py
    path = []
    i, j = max_x, max_y
    path.append((i, j))
    while i > 0 or j > 0:
        diag_cost = float('inf')
        left_cost = float('inf')
        down_cost = float('inf')

        if (i > 0) and (j > 0):
            diag_cost = D[i - 1][j - 1]
        if i > 0:
            left_cost = D[i - 1][j]
        if j > 0:
            down_cost = D[i][j - 1]

        if (diag_cost <= left_cost and diag_cost <= down_cost):
            i, j = i - 1, j - 1
        elif (left_cost < diag_cost and left_cost < down_cost):
            i = i - 1
        elif (down_cost < diag_cost and down_cost < left_cost):
            j = j - 1
        elif i <= j:
            j = j - 1
        else:
            i = i - 1
        path.append((i, j))
    path.reverse()
    return path

def calcDTWpath(a, b, metric='euclidean'):
    #print(a.shape)
    #print(b.shape)
    dm = cdist(a, b, metric=metric)
    mx, my = dm.shape
    path = backtrack(dm, mx - 1, my - 1)
    path_np = np.asarray(path)
    pathOf_a = path_np[:, 1]
    pathOf_b = path_np[:, 0]
    #print('path of a = ', pathOf_a)
    #print('path of b = ', pathOf_b)
    return pathOf_a, pathOf_b

def getCorrPath(a, b, a_frame_ids, b_frame_ids, metric='euclidean'):
    pa, pb = calcDTWpath(a, b, metric=metric)
    corrPath = np.vstack((a_frame_ids[pb].reshape(1,-1), b_frame_ids[pa].reshape(1,-1)))
    return corrPath

def plotConfMat(_confMat, labels, saveFileName='', iterID=-1, normalizeByAxis=-1, add2XLabel='', add2YLabel='', addCntXTicks=True, addCntYTicks=True, tickSize=8):
    if addCntXTicks:
        col_sums = _confMat.T.sum(axis=1)
        labelsX = labels.copy()
        for i in range(len(labels)):
            labelsX[i] = '({:d})-{:s}'.format(col_sums[i], labels[i])
    else:
        labelsX = labels.copy()
    if addCntYTicks:
        row_sums = _confMat.sum(axis=1)
        labelsY = labels.copy()
        for i in range(len(labels)):
            labelsY[i] = '({:d})-{:s}'.format(row_sums[i], labels[i])
    else:
        labelsY = labels.copy()

    if normalizeByAxis == 0:
        row_sums = _confMat.T.sum(axis=1)
        _confMat = (_confMat.T / row_sums[:, np.newaxis]).T
    elif normalizeByAxis == 1:
        row_sums = _confMat.sum(axis=1)
        _confMat = _confMat / row_sums[:, np.newaxis]

    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(_confMat)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels([''] + labelsX)
    ax.set_yticklabels([''] + labelsY)
    if iterID != -1:
        plt.xlabel('Predicted - iter {:03d}'.format(iterID + 1) + ' ' + add2XLabel)
    else:
        plt.xlabel('Predicted - ' + ' ' + add2XLabel)
    plt.ylabel('True' + ' ' + add2YLabel)
    plt.xticks(rotation=90, size=tickSize)
    plt.yticks(rotation=15, size=tickSize)
    if saveFileName=='':
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(saveFileName)

def sortVec(v):
    idx = np.unravel_index(np.argsort(-v, axis=None), v.shape)
    v = v[idx]
    return v, idx

def rnn_getTrainIDs(frameCnt, timesteps, frameOverlap, verbose=0):
    frSteps = timesteps - frameOverlap + 1
    blockCnt = 1 + np.ceil((frameCnt - timesteps) / frSteps).astype(int)
    fr = frSteps * np.asarray(range(blockCnt))
    to = fr + timesteps

    if (to[-1] > frameCnt):
        if verbose > 1:
            print(to[-1])
        to[-1] = frameCnt
        fr[-1] = frameCnt - timesteps

    if verbose > 1:
        ln = to - fr
        print(fr, "\n", to, "\n", ln)

    vidIDsTrain = []
    for i in range(len(to)):
        vidIDsTrain.append(np.arange(fr[i], to[i]))
    vidIDsTrain = np.array(vidIDsTrain).reshape(1, -1).squeeze()
    if verbose > 0:
        print('blockCnt = ', blockCnt)
        print('vidIDsTrain = ', vidIDsTrain)

    assertCond_1 = not np.any(vidIDsTrain > frameCnt - 1)
    assert assertCond_1, "no frameID can be bigger than frameCnt(" + str(frameCnt) + ")"
    assertCond_2 = vidIDsTrain[-1] == frameCnt - 1
    assert assertCond_2, "last frameID must be equal to frameCnt(" + str(frameCnt) + ")"

    return vidIDsTrain, blockCnt

def rnn_getValidIDs(frameCnt, timesteps, verbose=0):
    fr = timesteps*np.asarray(range(np.ceil(frameCnt/timesteps).astype(int)))
    to = np.asarray(fr + timesteps,dtype=int)
    if (to[-1]>frameCnt):
        to[-1] = frameCnt
        fr[-1] = frameCnt-timesteps

    vidIDsValid = []
    for i in range(len(to)):
        vidIDsValid.append(np.arange(fr[i], to[i]))
    vidIDsValid = np.array(vidIDsValid).reshape(1,-1).squeeze()
    part1 = np.arange(fr[0], fr[-1])
    part2 = np.arange(to[-2], len(vidIDsValid))
    frameIDsForLabelAcc = np.concatenate((part1, part2)).squeeze()
    if verbose>0:
        print('vidIDsValid = ', vidIDsValid)
        print('frameIDsForLabelAcc = ', frameIDsForLabelAcc)
    return vidIDsValid, frameIDsForLabelAcc

#mat = loadMatFile('/mnt/USB_HDD_1TB/neuralNetHandVideos_11/surfImArr_all_pca_256.mat')
def loadMatFile(matFileNameFull):
    mat = scipy.io.loadmat(matFileNameFull)
    print(sorted(mat.keys()))
    return mat

def getMappedKlusters(predictions, Kluster2ClassesK):
    uniqPredictions = np.unique(predictions)
    uniqClasses = np.unique(Kluster2ClassesK)

    mappedKlusters = np.copy(predictions)
    mappedKlustersSampleCnt = np.zeros([len(Kluster2ClassesK), len(uniqClasses)], dtype=int)

    for k in range(len(Kluster2ClassesK)):
        # cluster(k) will be mapped to Kluster2ClassesK[k]
        # predictions equal to k, will be mapped to Kluster2ClassesK[k]

        c = Kluster2ClassesK[k]  # the classID that kluster k will be mapped to

        slct = np.argwhere(predictions == k)  # predictions that are equal to k

        mappedKlusters[slct] = c  # map them to real classes

        mappedKlustersSampleCnt[k, c - 1] = len(slct)  # how many k's are mapped to c

    mappedKlustersSampleCnt = mappedKlustersSampleCnt.squeeze()
    return mappedKlusters, mappedKlustersSampleCnt

def analyzeClusterDistribution(predictedKlusters, n_clusters, verbose=0, printHistCnt=10):
    histOfClust, binIDs = np.histogram(predictedKlusters, np.unique(predictedKlusters))
    numOfBins = len(binIDs)
    numOf_1_sample_bins = np.sum(histOfClust==1)
    if verbose>0:
        print(n_clusters, " expected - ", numOfBins, " bins extracted. ", numOf_1_sample_bins, " of them have 1 sample")
    histSortedInv = np.sort(histOfClust)[::-1]
    if verbose>1:
        print("hist counts ascending = ", histSortedInv[0:printHistCnt])
    return numOf_1_sample_bins, histSortedInv

def getDict(retInds, retVals):
    ret_data = []
    for i in range(0, len(retInds)):
        ret_data.append([retInds[i], retVals[i]])
    return ret_data

def getPandasFromDict(retInds, retVals, columns):
    ret_data = getDict(retInds, retVals)
    if "pandas" in sys.modules:
        ret_data = pd_df(ret_data, columns=columns)
        print(ret_data)
    return ret_data

def calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=None):
    # print("This function gets predictions and real labels")
    # print("also a parameter that says either remove 0 labels or not")
    # print("an optional parameter as list of label names")
    np.unique(labels_true)
    # 1 remove zero labels if necessary

    # 2 calculate
    ars = metrics.adjusted_rand_score(labels_true, labels_pred)
    mis = metrics.mutual_info_score(labels_true, labels_pred)
    amis = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    hs = metrics.homogeneity_score(labels_true, labels_pred)
    cs = metrics.completeness_score(labels_true, labels_pred)
    vms = metrics.v_measure_score(labels_true, labels_pred)
    fms = metrics.fowlkes_mallows_score(labels_true, labels_pred)

    # 3 print
    retVals = [ars, mis, amis, cs, vms, fms, nmi, hs]
    retInds = ['adjusted_rand_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'completeness_score',
               'v_measure_score', 'fowlkes_mallows_score', 'normalized_mutual_info_score', 'homogeneity_score']
    ret_data = getPandasFromDict(retInds, retVals, columns=['metric', 'value'])

    return ret_data

def getInds(vec, val):
    return np.array([i for i, e in enumerate(vec) if e == val])

def getIndicedList(baseList, indsList):
    return [baseList[i] for i in indsList]

def get_most_frequent(List):
    return max(set(List), key = List.count)

def encode_sign_video_ID(signIDsOfSamples, videoIDsOfSamples):
    return signIDsOfSamples*1000 + videoIDsOfSamples

def decode_sign_video_ID(encodedIDs):
    videoIDsOfSamples = np.mod(encodedIDs, 1000)
    signIDsOfSamples = (encodedIDs-videoIDsOfSamples)/1000
    return signIDsOfSamples, videoIDsOfSamples

def getVideosToLabel(detailedLabels, labels_pred, predStr="", labelNames=None):
    # check detailedLabels according to labels_pred
    # for every kluster find the video to be labelled
    # repeat this procedure for every klusters 1, 2, 3 ... steps
    # for every step report the mapping of klusters and accuracy gained

    detailed_labels_obj, summaryInfoStr = generate_detailed_labels_obj(detailedLabels)
    print(summaryInfoStr)

    klusters_unique = np.unique(labels_pred)
    combinedSVIDs_all = encode_sign_video_ID(detailedLabels[:, 0], detailedLabels[:, 1])
    combinedSVIDs_unique = np.unique(combinedSVIDs_all)

    rCnt=len(klusters_unique)
    cCnt=len(combinedSVIDs_unique)
    cntMat = np.zeros([rCnt, cCnt], dtype=int)
    klustSampleCnts = np.zeros([rCnt, 1], dtype=int)
    for k in klusters_unique:
        # for every kluster find the video to be labelled
        klusterIndices = getInds(labels_pred, k)

        # videos are c0_c1 (s_v)
        signIDsOfSamples = detailedLabels[klusterIndices, 0]
        videoIDsOfSamples = detailedLabels[klusterIndices, 1]
        combinedSVIDs = encode_sign_video_ID(signIDsOfSamples, videoIDsOfSamples)

        k_ind = getInds(klusters_unique, k)
        klustSampleCnts[k_ind, 0] = len(klusterIndices)

        list_freq = (Counter(combinedSVIDs))
        for sv, cnt_cur in list_freq.items():
            sv_ind = getInds(combinedSVIDs_unique, sv)
            cntMat[k_ind, sv_ind] = cnt_cur

    pd_cntMat = pd_df(cntMat, columns=combinedSVIDs_unique, index=klusters_unique)

    iterID = -1
    vidLabelledCnt = 0
    vidToLabelList = []
    vidListsDict = []
    cntMat_Del = cntMat.copy()
    while vidLabelledCnt < cCnt:
        iterID = iterID + 1
        addList = []
        for k in klusters_unique:
            k_ind = getInds(klusters_unique, k)
            rBase = cntMat[k_ind, :].copy().squeeze()
            rCurr = list(cntMat_Del[k_ind, :].copy().squeeze())
            maxVal = max(rCurr)
            percVal = maxVal/np.sum(rBase)
            sv_ind = rCurr.index(maxVal)
            sv = combinedSVIDs_unique[sv_ind]
            cntMat_Del[k_ind, sv_ind] = 0
            if sv not in vidToLabelList:
                vidToLabelList.append(sv)
                vidLabelledCnt = vidLabelledCnt + 1
                addList.append([iterID, vidLabelledCnt, k, sv, percVal])
        vidListsDict.append([iterID, len(addList), np.sum(np.sum(cntMat_Del)), addList])

    signs_uniq = np.unique(detailedLabels[:,0])
    numOfSigns = len(signs_uniq)
    for s in signs_uniq:
        frIDs_s, lengths_s, labels_s = parse_detailed_labels_obj(detailed_labels_obj, s)
        labels_pred_sign = labels_pred[frIDs_s].reshape(-1, 1)


    return pd_cntMat

def calcPurity(labels_k):
    mappedClass = get_most_frequent(labels_k)
    try:
        correctLabelInds = getInds(labels_k, mappedClass)
    except:
        print("labels_k = ", labels_k)
        print("mappedClass = ", mappedClass)
        sys.exit("Error message")

    purity_k = 0
    if len(labels_k) > 0:
        purity_k = 100 * (len(correctLabelInds) / len(labels_k))

    return purity_k, correctLabelInds, mappedClass

def countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=None):
    sampleCount = labels_pred.size
    labels_pred2class = labels_pred.copy()
    kluster2Classes = []
    uniq_preds = np.unique(labels_pred)
    kr_data = []
    weightedPurity = 0
    cntUniqPred = uniq_preds.size
    for i in range(0, cntUniqPred):
        klust_cur = uniq_preds[i]
        inds = getInds(labels_pred, klust_cur)
        labels_k = getIndicedList(labels_true, inds)

        purity_k, correctLabelInds, mappedClass = calcPurity(labels_k)

        kluster2Classes.append([klust_cur, mappedClass, len(labels_k), correctLabelInds.size])
        labels_pred2class[inds] = mappedClass

        weightedPurity += purity_k * (len(inds) / sampleCount)

        try:
            cStr = "c(" + str(klust_cur) + ")" if labelNames is None else labelNames[mappedClass]
            #print('mappedClass {:d} : {}'.format(mappedClass, labelNames[mappedClass]))
        except:
            print("klust_cur = ", klust_cur)
            print("mappedClass = ", mappedClass)
            print("labelNames2.size = ", len(labelNames))
            sys.exit("Some indice error maybe")

        kr_data.append(["k" + str(uniq_preds[i]), cStr, len(correctLabelInds), len(inds), purity_k])

    kr_pdf = pd_df(kr_data, columns=['kID', 'mappedClass', '#of', 'N', '%purity'])
    kr_pdf.sort_values(by=['%purity', 'N'], inplace=True, ascending=[False, False])

    _confMat = confusion_matrix(labels_true, labels_pred2class)

    return _confMat, kluster2Classes

def calcCluster2ClassMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=None, predictDefStr=""):
    # print("This function gets predictions and real labels")
    # print("also a parameter that says either remove 0 labels or not")
    # print("an optional parameter as list of label names")
    # print("calc purity and assignment per cluster")

    # 4 map klusters to classes
    #_confMat, kluster2Classes = confusionFromKluster(labels_true, labels_pred)
    _confMat, kluster2Classes = countPredictionsForConfusionMat(labels_true, labels_pred)

    sampleCount = np.sum(np.sum(_confMat))
    acc = 100 * np.sum(np.diag(_confMat)) / sampleCount

    #kluster2Classes = kluster2Classes - 1
    print(predictDefStr, "-k2c", kluster2Classes)
    print("\r\n\r\n")

    kr_data = []
    uniq_preds = np.unique(labels_pred)
    weightedPurity = 0
    for i in range(0, uniq_preds.size):
        klust_cur = uniq_preds[i]
        mappedClass = kluster2Classes[i][1]

        inds = getInds(labels_pred, klust_cur)
        labels_k = getIndicedList(labels_true, inds)
        correctLabelInds = getInds(labels_k, mappedClass)
        purity_k = 0
        if len(labels_k) > 0:
            purity_k = 100 * (len(correctLabelInds) / len(labels_k))

        weightedPurity += purity_k * (len(inds) / sampleCount)

        cStr = "c(" + str(klust_cur) + ")" if labelNames is None else labelNames[mappedClass]
        kr_data.append(["k" + str(uniq_preds[i]), cStr, len(correctLabelInds), len(inds), purity_k])
    kr_pdf = pd_df(kr_data, columns=['kID', 'mappedClass', '#of', 'N', '%purity'])
    kr_pdf.sort_values(by=['%purity', 'N'], inplace=True, ascending=[False, False])

    analyzeClusterDistribution(labels_pred, max(uniq_preds), verbose=2, printHistCnt=len(uniq_preds))

    print(predictDefStr, "-kr_pdf:\r\n", kr_pdf, "\r\n\r\n")
    c_data = []
    uniq_labels = np.unique(labels_true)
    trueCnt = np.sum(_confMat, axis=1)
    predCnt = np.sum(_confMat, axis=0)
    weightedPrecision = 0
    weightedRecall = 0
    weightedF1Score = 0
    for i in range(0, uniq_labels.size):
        class_cur = uniq_labels[i]
        #mappedKlusters = getInds(kluster2Classes, class_cur)

        correctCnt = _confMat[class_cur, class_cur]
        if correctCnt==0:
            recallCur = 0
            precisionCur = 0
            f1Cur = 0
        else:
            recallCur = 100 * (correctCnt / trueCnt[class_cur])
            precisionCur = 100 * (correctCnt / predCnt[class_cur])
            f1Cur = 2 * ((precisionCur * recallCur) / (precisionCur + recallCur))
        if isNaN(f1Cur):
            print("****************************None")

        wp = precisionCur * (trueCnt[class_cur] / sampleCount)
        wr = recallCur * (trueCnt[class_cur] / sampleCount)
        wf = f1Cur * ((trueCnt[class_cur]) / (sampleCount))

        weightedPrecision += wp
        weightedRecall += wr
        weightedF1Score += wf

        cStr = ["c(" + class_cur + ")" if labelNames is None else labelNames[class_cur]]
        c_data.append([cStr, correctCnt, precisionCur, recallCur, f1Cur, wp, wr, wf])
    c_pdf = pd_df(c_data, columns=['class', '#', '%prec', '%recall', '%f1', '%wp', '%wr', '%wf'])
    c_pdf.sort_values(by=['%f1', '#'], inplace=True, ascending=[False, False])

    retVals = [acc, weightedPurity, weightedPrecision, weightedRecall, weightedF1Score]
    retInds = ['accuracy', 'weightedPurity', 'weightedPrecision', 'weightedRecall', 'weightedF1Score']
    classRet = getPandasFromDict(retInds, retVals, columns=['metric', 'value'])

    print(predictDefStr, "-c_pdf:\r\n", c_pdf, "\r\n\r\n")
    # print("classRet:\r\n",json.dumps(classRet, indent = 2),"\r\n\r\n")
    plotConfMat(_confMat, labelNames, addCntXTicks=False, addCntYTicks=False, tickSize=10)
    print(predictDefStr, "-_confMat:\r\n", pd_df(_confMat.T, columns=labelNames, index=labelNames), "\r\n\r\n")

    return classRet, _confMat, c_pdf, kr_pdf

def generate_detailed_labels_obj(detailedLabels):
    # detailedLabels ->[signID videoId frameID labelOfFrame]
    sList = np.array(np.unique(detailedLabels[:, 0]), dtype=int)
    fr = 0
    detailed_labels_obj = []
    summaryInfoStr = ""
    for s in sList:
        detailedLabels_sign_rows = np.argwhere(detailedLabels[:, 0] == s).flatten()
        to = fr + len(detailedLabels_sign_rows)
        summaryInfoStr += "sign({:d}),frameCnt({:d}),fr({:d}),to({:d})\r\n".format(s, to - fr, fr, to)
        fr = to
        detailedLabels_sign = detailedLabels[detailedLabels_sign_rows, :]
        # print(detailedLabels_sign.shape)
        vList = np.array(np.unique(detailedLabels_sign[:, 1]), dtype=int)
        # print(vList.shape)
        vD = []
        for v in vList:
            detailedLabels_video_rows = np.argwhere(detailedLabels_sign[:, 1] == v).flatten()
            detailedLabels_video = detailedLabels_sign[detailedLabels_video_rows, :]

            videoLabels = detailedLabels_video[:, 3]

            frIDs = detailedLabels_sign_rows[detailedLabels_video_rows]

            vD.append({"vID": v, "labels": videoLabels, "frIDs": frIDs})
        detailed_labels_obj.append({"sID": s, "videoDict": vD})
    return detailed_labels_obj, summaryInfoStr

def parse_detailed_labels_obj(detailed_labels_obj, signID):
    frIDs = []
    lengths = []
    labels = []
    for s in detailed_labels_obj:
        if signID == s["sID"]:
            vD = s["videoDict"]
            for v in vD:
                vID = v["vID"]
                videoLabels = v["labels"]
                videoFrameIDs = v["frIDs"]

                labels = np.concatenate([labels, videoLabels])
                frIDs = np.concatenate([frIDs, videoFrameIDs])
                lengths.append(len(videoFrameIDs))

    labels = np.array(labels, dtype=int)
    frIDs = np.array(frIDs, dtype=int)
    lengths = np.array(lengths, dtype=int)
    print("signID:{:d}, frIDs.shape:{}, lengths.shape:{}, labels.shape:{}".format(signID, frIDs.shape, lengths.shape, labels.shape))
    return frIDs, lengths, labels