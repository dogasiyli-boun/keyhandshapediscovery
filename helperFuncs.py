import socket
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.spatial.distance import cdist

def getVariableByComputerName(variableName):
    curCompName = socket.gethostname()
    if variableName=='base_dir':
        if curCompName == 'doga-MSISSD':
            base_dir = '/mnt/USB_HDD_1TB'  # for bogazici kasa
        else:
            base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = base_dir
    if variableName=='data_dir':
        if curCompName == 'doga-MSISSD':
            data_dir = '/mnt/USB_HDD_1TB/bdData'  # for bogazici kasa
        else:
            data_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        retVal = data_dir
    if variableName=='results_dir':
        if curCompName == 'doga-MSISSD':
            results_dir = '/mnt/USB_HDD_1TB/bdResults'  # for bogazici kasa
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
    rowLabels, colLabels = createLabelsForConfMat(inputConfMat)
    acc = np.sum(np.diag(confMat)) / np.sum(np.sum(confMat))
    # nmiAr  = nmi(colLabels,rowLabels,average_method='arithmetic')
    # nmiGeo = nmi(colLabels,rowLabels,average_method='geometric')
    return acc
    # print('confAcc(', acc ,'),nmiAr(', nmiAr,')nmiGeo(',nmiGeo ,')')

def get_NMI_Acc(non_zero_labels, non_zero_predictions):
    nmi_cur = nmi(non_zero_labels, non_zero_predictions, average_method='geometric')
    acc_cur = getAccFromConf(non_zero_labels, non_zero_predictions)
    return nmi_cur, acc_cur

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