import socket
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from sklearn.metrics import normalized_mutual_info_score as nmi

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