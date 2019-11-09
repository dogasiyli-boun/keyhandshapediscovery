import socket
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_sc

from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def printMatRes(X, defStr):
    print(defStr, '\n', X)
    print('samples-rows : ', X.shape[0], 'feats-cols : ', X.shape[1])
    mean0_X = np.mean(X, axis=0)
    mean1_X = np.mean(X, axis=1)
    print('mean over samples,axis=0', mean0_X, ', cnt=', len(mean0_X))
    print('correct --> mean over features,axis=1', mean1_X, ', cnt=', len(mean1_X))

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

def getVariableByComputerName(variableName):
    curCompName = socket.gethostname()
    if variableName=='base_dir':
        if curCompName == 'doga-MSISSD':
            base_dir = '/mnt/USB_HDD_1TB'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            base_dir = '/home/dg/DataPath'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            base_dir = '/home/doga/DataFolder'  # for laptop
        else:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = base_dir
    if variableName=='data_dir':
        if curCompName == 'doga-MSISSD':
            data_dir = '/mnt/USB_HDD_1TB/bdData'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            data_dir = '/home/dg/DataPath/bdData'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            data_dir = '/home/doga/DataFolder/bdData'  # for laptop
        else:
            data_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = data_dir
    if variableName=='results_dir':
        if curCompName == 'doga-MSISSD':
            results_dir = '/mnt/USB_HDD_1TB/bdResults'  # for bogazici kasa
        elif curCompName == 'WsUbuntu05':
            results_dir = '/home/dg/DataPath/bdResults'  # for WS Doga DHO
        elif curCompName == 'doga-msi-ubu':
            results_dir = '/home/doga/DataFolder/bdResults'  # for laptop
        else:
            results_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = results_dir
    return retVal

def createDirIfNotExist(dir2create):
    if not os.path.isdir(dir2create):
        os.makedirs(dir2create)

def normalize(x, axis=0):
    s = np.sum(x, axis=axis, keepdims=True)
    return x / s

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

def get_NMI_Acc(non_zero_labels, non_zero_predictions):
    nmi_cur = nmi(non_zero_labels, non_zero_predictions, average_method='geometric')
    acc_cur = getAccFromConf(non_zero_labels, non_zero_predictions)
    return nmi_cur, acc_cur

def applyMatTransform(featVec, applyNormalization=True, applyPca=True, whiten=True):
    if applyPca:
        pca = PCA(whiten=whiten, svd_solver='full')
        featVec = pca.fit_transform(featVec)
    if applyNormalization:
        featVec = featVec / np.linalg.norm(featVec)
    return featVec
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

def getNonZeroLabels(labVec, predictedKlusters):
    labVec = np.asarray(labVec, dtype=int)
    predictedKlusters = np.asarray(predictedKlusters, dtype=int)
    predictedKlusters = predictedKlusters[np.where(labVec)]
    labVec = labVec[np.where(labVec)]
    return labVec, predictedKlusters

def get_nmi(featVec, labVec, n_clusters, applyNormalization=True, applyPca=True):
    featVec = applyMatTransform(np.array(featVec), applyNormalization, applyPca)
    df = DataFrame(featVec)

    kmeans_result = KMeans(n_clusters=n_clusters).fit(df)
    predictedKlusters = kmeans_result.labels_.astype(float)
    nmi_score = nmi_sc(labVec, predictedKlusters, average_method='geometric')

    labVec_nonzero, predictedKlusters_nonzero = getNonZeroLabels(labVec, predictedKlusters)
    nmi_score_nonzero = nmi_sc(labVec_nonzero, predictedKlusters_nonzero, average_method='geometric')

    return nmi_score, predictedKlusters, nmi_score_nonzero

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

def plotConfMat(_confMat, labels, saveFileName='', iterID=-1, normalizeByAxis=-1, add2XLabel='', add2YLabel='', addCntXTicks=True, addCntYTicks=True):
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
    plt.xticks(rotation=90, size=8)
    plt.yticks(rotation=15, size=8)
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

def loadCorrespondantFrames(corrFramesFileNameFull):
    corrFramesAll = np.load(corrFramesFileNameFull)
    return corrFramesAll

