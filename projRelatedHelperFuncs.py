import helperFuncs as funcH
import dataLoaderFuncs as funcD
import ensembleFuncs as funcEns
import visualize as funcVis
import numpy as np
import os
from numpy.random import seed
import tensorflow as tf
import pandas as pd
from pandas import DataFrame as pd_df
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.metrics import confusion_matrix
from collections import Counter
from clusteringWrapper import Clusterer

from torch.utils.data import DataLoader
import csv

from matplotlib.ticker import FormatStrFormatter

def createExperimentName(trainParams, modelParams, rnnParams):

    pcaCountStr = str(modelParams["pcaCount"]) if modelParams["pcaCount"] > 0 else "Feats"
    normModeStr = str(modelParams["normMode"]) if modelParams["normMode"] == "" else "_" + str(modelParams["normMode"]) + "_"

    allInitStr = str(modelParams["trainMode"]) + \
                 '_pd' + str(modelParams["posterior_dim"]) + \
                 '_wr' + str(modelParams["weight_of_regularizer"]) + \
                 '_' + str(modelParams["dataToUse"]) + normModeStr + pcaCountStr + '_' + str(modelParams["numOfSigns"]) + \
                 '_bs' + str(trainParams["batch_size"]) + \
                 '_rs' + str(trainParams["randomSeed"])

    if modelParams["trainMode"] == "corsa":
        exp_name  = allInitStr + \
                    '_dM' + str(rnnParams["dataMode"]) + \
                    '_ts' + str(rnnParams["timesteps"]) + \
                    '_cp' + str(trainParams["applyCorr"]) + \
                    '_cRM' + str(int(trainParams["corr_randMode"])) + \
                    '_cSM' + str(int(trainParams["corr_swapMode"]))
        if rnnParams["dropout"] > 0:
            exp_name += '_do' + str(rnnParams["dropout"])
    elif modelParams["trainMode"] == "rsa":
        exp_name  = allInitStr + \
                    '_dM' + str(rnnParams["dataMode"]) + \
                    '_ts' + str(rnnParams["timesteps"])
        if rnnParams["dropout"] > 0:
            exp_name += '_do' + str(rnnParams["dropout"])
        if rnnParams["dataMode"] == 1:
            exp_name += '_pc' + str(rnnParams["patchFromEachVideo"])
        if rnnParams["dataMode"] == 2:
            exp_name += '_fo' + str(rnnParams["frameOverlap"])
    elif modelParams["trainMode"] == "cosae":
        exp_name  = allInitStr + \
                    '_cp' + str(trainParams["applyCorr"]) + \
                    '_cRM' + str(int(trainParams["corr_randMode"])) + \
                    '_cSM' + str(int(trainParams["corr_swapMode"]))
    elif modelParams["trainMode"] == "sae":
        exp_name  = allInitStr
    return exp_name

def createExperimentDirectories(results_dir, exp_name):
    csv_name = os.path.join(results_dir, 'epochs', exp_name + '.csv')
    model_name = os.path.join(results_dir, 'models', exp_name + '.h5')
    outdir = os.path.join(results_dir, 'results', exp_name)
    funcH.createDirIfNotExist(os.path.join(results_dir, 'epochs'))
    funcH.createDirIfNotExist(os.path.join(results_dir, 'models'))
    funcH.createDirIfNotExist(outdir)
    return csv_name, model_name, outdir

#features, labels = getFeatsFromMat(mat,'dataCurdim', 'labelVecs_all')
def getFeatsFromMat(mat, featureStr, labelStr):
    features = np.asarray(mat[featureStr], dtype=float)
    labels = np.asarray(mat[labelStr], dtype=int)
    return features, labels

def getMatFile(data_dir, signCnt, possible_fname_init):
    # possible_fname_init = ['surfImArr', 'snFeats']
    # possible_fname_init = ['skeleton', 'skelFeats']
    matFileName_1 = os.path.join(data_dir, possible_fname_init[0] + '_' + str(signCnt) + '.mat')
    matFileName_2 = os.path.join(data_dir, possible_fname_init[1] + '_' + str(signCnt) + '.mat')
    if os.path.isfile(matFileName_1):
        matFileName = matFileName_1
    elif os.path.isfile(matFileName_2):
        matFileName = matFileName_2
    else:
        print('neither ', matFileName_1, ' nor ', matFileName_2, ' is a file')
        return []
    print(matFileName, ' is loaded :) ')
    mat = funcH.loadMatFile(matFileName)
    return mat

def createPCAOfData(data_dir, dataToUse, sign_count, recreate=False, normMode=''):
    # normMode = str(modelParams["normMode"])
    npy_PCAFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=-1, numOfSigns=sign_count, expectedFileType='PCA')
    npy_PCAFileName = os.path.join(data_dir, npy_PCAFileName)
    if os.path.isfile(npy_PCAFileName) and not recreate:
        feats_pca = np.load(npy_PCAFileName)
        exp_var_rat = []
        print('loaded ', dataToUse, 'Feats(', feats_pca.shape, ') from : ', npy_PCAFileName)
        print('Max of featsPCA = ', np.amax(feats_pca), ', Min of featsPCA = ', np.amin(feats_pca))
    else:
        npy_FeatsFileName = funcD.getFileName(dataToUse=dataToUse, normMode='', pcaCount=-1, numOfSigns=sign_count, expectedFileType='Data')
        npy_FeatsFileName = os.path.join(data_dir, npy_FeatsFileName)
        feats = np.load(npy_FeatsFileName)
        print('Max of feats = ', np.amax(feats), ', Min of feats = ', np.amin(feats))
        feats_pca, exp_var_rat = funcH.applyMatTransform(feats, applyPca=True, whiten=True, normMode=normMode, verbose=2)
        np.save(npy_PCAFileName, feats_pca)
    return feats_pca, exp_var_rat

def convert_Mat2NPY(dataToUse, data_dir, signCnt, featureStr, labelStr, possible_fname_init, recreate=False):
    npy_labels_file_name = funcD.getFileName(dataToUse=dataToUse, normMode='', pcaCount=-1, numOfSigns=signCnt, expectedFileType='Labels')
    npy_labels_file_name = os.path.join(data_dir, npy_labels_file_name)
    npy_feats_file_name = funcD.getFileName(dataToUse=dataToUse, normMode='', pcaCount=-1, numOfSigns=signCnt, expectedFileType='Data')
    npy_feats_file_name = os.path.join(data_dir, npy_feats_file_name)

    if not recreate and os.path.isfile(npy_labels_file_name) and os.path.isfile(npy_feats_file_name):
        labels = np.load(npy_labels_file_name)
        print('loaded labels(', labels.shape, ') from : ', npy_labels_file_name)
        feats = np.load(npy_feats_file_name)
        print('loaded ', dataToUse, 'Feats(', feats.shape, ') from : ', npy_feats_file_name)
    else:
        mat_feats = getMatFile(data_dir, signCnt, possible_fname_init)
        feats, labels = getFeatsFromMat(mat_feats, featureStr, labelStr)

        labels = np.reshape(labels, (len(labels), -1)).squeeze()
        print('saving labels(', labels.shape, ') at : ', npy_labels_file_name)
        np.save(npy_labels_file_name, labels)
        print('saving ', dataToUse, 'Feats(', feats.shape, ') at : ', npy_feats_file_name)
        np.save(npy_feats_file_name, feats)
    return feats, labels

def createPCADimsOfData(data_dir, data2use, sign_count, dimArray = [256, 512, 1024], recreate=False, normMode=''):
    featsPCA, exp_var_rat = createPCAOfData(data_dir=data_dir, dataToUse=data2use, sign_count=sign_count, recreate=recreate, normMode=normMode)
    try:
        print(data2use, ' feats exp_var_rat[', list(map('{:.2f}%'.format, dimArray)), '] = ', list(map('{:.2f}%'.format, exp_var_rat[dimArray])), ', normMode=<', str(normMode), '>')
    except:
        print(data2use, ' feats exp_var_rat[', list(map('{:.2f}%'.format, dimArray)), '], normMode=<', str(normMode), '>')
        pass

    for dims in dimArray:
        npy_PCAFileName = funcD.getFileName(dataToUse=data2use, normMode=normMode, pcaCount=dims, numOfSigns=sign_count, expectedFileType='Data')
        npy_PCAFileName = os.path.join(data_dir, npy_PCAFileName)
        featsToSave = featsPCA[:,0:dims]
        if os.path.isfile(npy_PCAFileName) and not recreate:
            featsToSave = np.load(npy_PCAFileName)
            print('pca exists at : ', npy_PCAFileName)
        else:
            print("features.shape:", featsToSave.shape)
            print('saving pca sn features at : ', npy_PCAFileName)
            np.save(npy_PCAFileName, featsToSave)

def runClusteringOnFeatSet(data_dir, results_dir, dataToUse, normMode, numOfSigns, pcaCount, expectedFileType, clustCntVec = [64, 128, 256], clusterModels = ['KMeans', 'GMM_diag', 'GMM_full', 'Spectral'], randomSeed=5):
    seed(randomSeed)
    try:
        tf.set_random_seed(seed=randomSeed)
    except:
        tf.random.set_seed(seed=randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)

    featsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType=expectedFileType) # 'hogFeats_41.npy', 'skeletonFeats_41.npy'
    detailedLabelsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='DetailedLabels') # 'detailedLabels_41.npy'
    detailedLabelsFileNameFull = data_dir + os.sep + detailedLabelsFileName
    labelsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='Labels') # 'labels_41.npy'
    labelsFileNameFull = data_dir + os.sep + labelsFileName
    labelNames = load_label_names(numOfSigns)

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    funcH.createDirIfNotExist(os.path.join(results_dir, 'baseResults'))
    baseResultFileNameFull = os.path.join(results_dir, 'baseResults', baseResultFileName)

    featSet = np.load(data_dir + os.sep + featsFileName)
    detailedLabels_all = np.load(detailedLabelsFileNameFull)
    labels_all = np.load(labelsFileNameFull)
    non_zero_labels = labels_all[np.where(labels_all)]

    print('*-*-*-*-*-*-*running for : ', featsFileName, '*-*-*-*-*-*-*')
    print('featSet(', featSet.shape, '), detailedLabels(', detailedLabels_all.shape, '), labels_All(', labels_all.shape, '), labels_nonzero(', non_zero_labels.shape, ')')

    # clustCntVec = [64, 128, 256] #[32, 64, 128, 256, 512]
    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = list(np.load(baseResultFileNameFull, allow_pickle=True))
    else:
        resultDict = []

    headerStrFormat = "+++frmfile(%15s) clusterModel(%8s), clusCnt(%4s)"
    valuesStrFormat = "nmiAll(%.2f) * accAll(%.2f) * nmiNoz(%.2f) * accNoz(%.2f) * emptyClusters(%d) * meanPurity(%5.3f) * weightedPurity(%5.3f)"

    for clusterModel in clusterModels:
        for curClustCnt in clustCntVec:
            foundResult = False
            for resultList in resultDict:
                if resultList[1] == clusterModel and resultList[2] == curClustCnt:
                    str2disp = headerStrFormat + "=" + valuesStrFormat
                    data2disp = (baseResultFileName, resultList[1], resultList[2],
                                 resultList[3][0], resultList[3][1], resultList[3][2], resultList[3][3], resultList[3][4], resultList[3][5], resultList[3][6])
                    #histCnt=', resultList[3][5][0:10])
                    print(str2disp % data2disp)
                    foundResult = True
                if foundResult:
                    break
            predictionFileName = baseResultFileName.replace("_baseResults.npy","") + "_" + clusterModel + "_" + str(curClustCnt) + ".npz"
            predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
            predictionFileExist = os.path.isfile(predictionFileNameFull)
            if not foundResult or not predictionFileExist:
                if foundResult and not predictionFileExist:
                    print('running again for saving predictions')
                t = time.time()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(featsFileName, 'clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ') running.')
                predClusters, _ = Clusterer(cluster_model=clusterModel, n_clusters=curClustCnt).fit_predict(X=featSet, post_analyze_distribution=True, verbose=1)
                print('elapsedTime(', time.time() - t, ')')

                nmi_cur, acc_cur = funcH.get_NMI_Acc(labels_all, predClusters)

                non_zero_preds = predClusters[np.where(labels_all)]
                nmi_cur_nz, acc_cur_nz = funcH.get_NMI_Acc(non_zero_labels, non_zero_preds)

                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)
                _, _, kr_pdf, weightedPurity = funcH.countPredictionsForConfusionMat(non_zero_labels, non_zero_preds, labelNames=labelNames)
                meanPurity = np.mean(np.asarray(kr_pdf["%purity"]))

                resultList = [featsFileName.replace('.npy', ''), clusterModel, curClustCnt, [nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, meanPurity, weightedPurity, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)

                print(valuesStrFormat % (nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins, meanPurity, weightedPurity))
                print(resultList[3][5][0:10])
                np.save(baseResultFileNameFull, resultDict, allow_pickle=True)
                np.savez(predictionFileNameFull, labels_all, predClusters)

    np.set_printoptions(prevPrintOpts)
    return resultDict

def runClusteringOnFeatSet_Aug2020(ft, labels_all, lb_map, dataToUse, numOfSigns, pcaCount, clustCntVec = None, clusterModels = ['KMeans'], randomSeed=5):
    seed(randomSeed)
    try:
        tf.set_random_seed(seed=randomSeed)
    except:
        tf.random.set_seed(seed=randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)
    labels_all = np.squeeze(np.array(labels_all))
    class_names = np.asarray(lb_map["khsName"])

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    funcH.createDirIfNotExist(os.path.join(results_dir, 'baseResults'))
    baseResultFileNameFull = os.path.join(results_dir, 'baseResults', baseResultFileName)

    featsName = baseResultFileName.replace(".npy", "")  # '<dataToUse><pcaCount>_baseResults_<nos>.npy'

    print('*-*-*-*-*-*-*running for : ', featsName, '*-*-*-*-*-*-*')
    print('featSet(', ft.shape, '), labels_All(', labels_all.shape, ')')

    if clustCntVec is None:
        if ft.shape[1] > 128:
            clustCntVec = [128, 256, 512]
        else:
            clustCntVec = [32, 64, 96]
    # clustCntVec = [64, 128, 256] #[32, 64, 128, 256, 512]
    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = list(np.load(baseResultFileNameFull, allow_pickle=True))
    else:
        resultDict = []

    headerStrFormat = "+++frmfile(%15s) clusterModel(%8s), clusCnt(%4s)"
    valuesStrFormat = "nmiAll(%.2f) * acc_cent(%.2f) * meanPurity_cent(%.3f) * weightedPurity_cent(%.3f) * acc_mxhs(%.2f) * meanPurity_mxhs(%.3f) * weightedPurity_mxhs(%.3f) * cnmxh_perc(%.3f) * emptyClusters(%d)"

    for clusterModel in clusterModels:
        for curClustCnt in clustCntVec:
            foundResult = False
            for resultList in resultDict:
                if resultList[1] == clusterModel and resultList[2] == curClustCnt:
                    str2disp = headerStrFormat + "=" + valuesStrFormat
                    data2disp = (baseResultFileName, resultList[1], resultList[2],
                                 resultList[3][0],
                                 resultList[3][1], resultList[3][2], resultList[3][3],
                                 resultList[3][4], resultList[3][5], resultList[3][6], resultList[3][7],
                                 resultList[3][8])
                    #histCnt=', resultList[3][9][0:10])
                    print(str2disp % data2disp)
                    foundResult = True
                if foundResult:
                    break
            predictionFileName = baseResultFileName.replace("_baseResults.npy","") + "_" + clusterModel + "_" + str(curClustCnt) + ".npz"
            predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
            predictionFileExist = os.path.isfile(predictionFileNameFull)
            if not foundResult or not predictionFileExist:
                if foundResult and not predictionFileExist:
                    print('running again for saving predictions')
                t = time.time()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(featsName, 'clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ') running.')
                predClusters, centroid_info_pdf = Clusterer(cluster_model=clusterModel, n_clusters=curClustCnt).fit_predict(X=ft, post_analyze_distribution=True, verbose=1)
                print('elapsedTime(', funcH.getElapsedTimeFormatted(time.time() - t), ')')
                #np.savez("/home/doga/DataFolder/bdResults/baseResults/prdclu.npz", predClusters=predClusters, kluster_centers=kluster_centers)

                nmi_cur = 100*funcH.get_nmi_only(labels_all, predClusters)
                _confMat_mapped_preds_center, _, kr_pdf_center, weightedPurity_center, cnmxh_perc = funcH.countPredictionsForConfusionMat(labels_all, predClusters, labelNames=class_names, centroid_info_pdf=centroid_info_pdf, verbose=0)
                _confMat_mapped_preds_mxhist, _, kr_pdf_mxhist, weightedPurity_mxhist, _ = funcH.countPredictionsForConfusionMat(labels_all, predClusters, labelNames=class_names, centroid_info_pdf=None, verbose=0)
                acc_center = 100*np.sum(np.diag(_confMat_mapped_preds_center)) / np.sum(np.sum(_confMat_mapped_preds_center))
                acc_mxhist = 100*np.sum(np.diag(_confMat_mapped_preds_mxhist)) / np.sum(np.sum(_confMat_mapped_preds_mxhist))
                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)
                meanPurity_center = np.mean(np.asarray(kr_pdf_center["%purity"]))
                meanPurity_mxhist = np.mean(np.asarray(kr_pdf_mxhist["%purity"]))

                resultList = [featsName, clusterModel, curClustCnt, [nmi_cur, acc_center, meanPurity_center, weightedPurity_center, acc_mxhist, meanPurity_mxhist, weightedPurity_mxhist, cnmxh_perc, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)

                print(valuesStrFormat % (nmi_cur,  acc_center, meanPurity_center, weightedPurity_center, acc_mxhist, meanPurity_mxhist, weightedPurity_mxhist, cnmxh_perc, numOf_1_sample_bins))
                print("histogram of clusters max first 10", resultList[3][9][0:10])
                np.save(baseResultFileNameFull, resultDict, allow_pickle=True)
                np.savez(predictionFileNameFull, labels_all, predClusters)

    np.set_printoptions(prevPrintOpts)
    return resultDict

def runOPTICSClusteringOnFeatSet(data_dir, results_dir, dataToUse, normMode, pcaCount, numOfSigns, expectedFileType, clustCntVec = [32, 64, 128, 256, 512], randomSeed=5, updateResultBaseFile=False):
    seed(randomSeed)
    tf.set_random_seed(seed=randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)

    featsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType=expectedFileType) # 'hogFeats_41.npy', 'skeletonFeats_41.npy'
    detailedLabelsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='DetailedLabels') # 'detailedLabels_41.npy'
    detailedLabelsFileNameFull = data_dir + os.sep + detailedLabelsFileName
    labelsFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='Labels') # 'labels_41.npy'
    labelsFileNameFull = data_dir + os.sep + labelsFileName

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    funcH.createDirIfNotExist(os.path.join(results_dir, 'baseResults'))
    baseResultFileNameFull = os.path.join(results_dir, 'baseResults', baseResultFileName)

    featSet = np.load(data_dir + os.sep + featsFileName)
    detailedLabels_all = np.load(detailedLabelsFileNameFull)
    labels_all = np.load(labelsFileNameFull)
    non_zero_labels = labels_all[np.where(labels_all)]

    print('*-*-*-*-*-*-*running for : ', featsFileName, '*-*-*-*-*-*-*')
    print('featSet(', featSet.shape, '), detailedLabels(', detailedLabels_all.shape, '), labels_All(', labels_all.shape, '), labels_nonzero(', non_zero_labels.shape, ')')

    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = list(np.load(baseResultFileNameFull, allow_pickle=True))
    else:
        resultDict = []

    headerStrFormat = "+++frmfile(%15s) clusterModel(%8s), clusCnt(%4s)"
    valuesStrFormat = "nmiAll(%.2f) * accAll(%.2f) * nmiNoz(%.2f) * accNoz(%.2f) * emptyClusters(%d)"

    clusterModels = []  # pars = clusterModel.split('_')  # 'OPTICS_hamming_dbscan', 'OPTICS_russellrao_xi'
    metricsAvail = np.sort(
        ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
         'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
         'sokalsneath', 'sqeuclidean', 'yule',
         'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
    cluster_methods_avail = ['xi', 'dbscan']
    for metric in metricsAvail:
        for cm in cluster_methods_avail:
            clusterModels.append('OPTICS_' + metric + '_' + cm)

    for clusterModel in clusterModels:
        for curClustCnt in clustCntVec:
            foundResult = False
            for resultList in resultDict:
                if resultList[1] == clusterModel and resultList[2] == curClustCnt:
                    str2disp = headerStrFormat + "=" + valuesStrFormat
                    data2disp = (baseResultFileName, resultList[1], resultList[2],
                                 resultList[3][0], resultList[3][1], resultList[3][2], resultList[3][3], resultList[3][4])
                    #histCnt=', resultList[3][5][0:10])
                    print(str2disp % data2disp)
                    foundResult = True
                if foundResult:
                    break
            predictionFileName = baseResultFileName.replace("_baseResults.npy", "") + "_" + clusterModel + "_" + str(curClustCnt) + ".npz"
            predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
            predictionFileExist = os.path.isfile(predictionFileNameFull)
            if not predictionFileExist:  # not foundResult or
                #if foundResult and not predictionFileExist:
                #    print('running again for saving predictions')
                t = time.time()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(featsFileName, 'clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ') running.')
                predClusters, _ = Clusterer(cluster_model=clusterModel, n_clusters=curClustCnt).fit_predict(X=featSet, post_analyze_distribution=True, verbose=1)
                print('elapsedTime(', time.time() - t, ')')
                np.savez(predictionFileNameFull, labels_all, predClusters)
            else:
                npz = np.load(predictionFileNameFull)
                if 'arr_0' in npz.files:
                    np.savez(predictionFileNameFull, labels_all=npz['arr_0'], predClusters=npz['arr_1'])
                    npz = np.load(predictionFileNameFull)
                predClusters = npz['predClusters']

            if not foundResult:
                nmi_cur, acc_cur = funcH.get_NMI_Acc(labels_all, predClusters)

                non_zero_preds = predClusters[np.where(labels_all)]
                nmi_cur_nz, acc_cur_nz = funcH.get_NMI_Acc(non_zero_labels, non_zero_preds)

                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)

                resultList = [featsFileName.replace('.npy', ''), clusterModel, curClustCnt, [nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)

                print(valuesStrFormat % (nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins))
                print(resultList[3][5][0:10])
                if updateResultBaseFile:
                    np.save(baseResultFileNameFull, resultDict, allow_pickle=True)

    np.set_printoptions(prevPrintOpts)
    return resultDict

def load_label_names(nos):
    data_dir = funcH.getVariableByComputerName('data_dir')
    if nos == 8:
        labelnames_csv_filename = os.path.join(data_dir, "khsList_33_41_19.csv")
    elif nos == 10:
        labelnames_csv_filename = os.path.join(data_dir, "khsList_23_33_26.csv")
    elif nos == 11:
        labelnames_csv_filename = os.path.join(data_dir, "khsList_0_11_26.csv")
    elif nos == 12:
        labelnames_csv_filename = os.path.join(data_dir, "khsList_11_23_30.csv")
    else:
        os.error(nos)
    labelNames = list(pd.read_csv(labelnames_csv_filename, sep=",")['name'].values.flatten())
    return labelNames

def loadData(model_params, numOfSigns, data_dir):
    featsFileName = funcD.getFileName(dataToUse=model_params["dataToUse"], normMode=str(model_params["normMode"]), pcaCount=model_params["pcaCount"],
                                      numOfSigns=numOfSigns, expectedFileType='Data')
    fileName_detailedLabels = funcD.getFileName(dataToUse=model_params["dataToUse"], normMode=str(model_params["normMode"]), pcaCount=model_params["pcaCount"],
                                                numOfSigns=numOfSigns, expectedFileType='DetailedLabels')
    fileName_labels = funcD.getFileName(dataToUse=model_params["dataToUse"], normMode=str(model_params["normMode"]), pcaCount=model_params["pcaCount"],
                                        numOfSigns=numOfSigns, expectedFileType='Labels')

    feat_set = funcD.loadFileIfExist(data_dir, featsFileName)
    if feat_set.size == 0:
        feats_pca, exp_var_rat = createPCAOfData(data_dir, dataToUse=model_params["dataToUse"], sign_count=numOfSigns, recreate=False, normMode=str(model_params["normMode"]))
        feat_set = feats_pca[:, 0:model_params["pcaCount"]]
        np.save(os.path.join(data_dir, featsFileName), feat_set)
        feat_set = funcD.loadFileIfExist(data_dir, featsFileName)
        if feat_set.size == 0:
            os.error("finish here")
        # apply needed transofrmation and use the data


    detailed_labels_all = funcD.loadFileIfExist(data_dir, fileName_detailedLabels)
    labels_all = funcD.loadFileIfExist(data_dir, fileName_labels)

    return feat_set, labels_all, detailed_labels_all

def loadBaseResult(fileName):
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    if not str(fileName).endswith(".npz"):
        fileName = fileName + ".npz"
    fileName = os.path.join(baseLineResultFolder, fileName)
    #print("fileName=", fileName)
    preds = np.load(fileName)
    labels_true = np.asarray(preds['arr_0'], dtype=int)
    labels_pred = np.asarray(preds['arr_1'], dtype=int)
    return labels_true, labels_pred

def getBaseResults(dataToUse, normMode, pcaCount, numOfSigns, displayResults=True, baseResultFileName=''):
    if baseResultFileName == '':
        baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount,
                                               numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResultFileNameFull = os.path.join(baseLineResultFolder, baseResultFileName)

    resultDict = np.load(baseResultFileNameFull, allow_pickle=True)
    headerStrFormatBase = "%15s * %10s * %9s "
    headerStrFormat = headerStrFormatBase + "* %6s * %6s * %6s * %6s * %6s"
    #valuesStrFormat = headerStrFormatBase + "* nmiAll(%6.2f) * accAll(%6.2f) * nmiNoz(%6.2f) * accNoz(%6.2f) * emptyClusters(%6d)"
    valuesStrFormat2= headerStrFormatBase + "* %6.2f * %6.2f * %6.2f * %6.2f * %6d"

    #print(headerStrFormat % ("npyFileName", "clusModel", "clusCnt", "nmiAll", "accAll", "nmiNoz", "accNoz", "emptyK"))
    #baseResults = {}
    #baseResultsLab = []
    #baseResultsVal = []
    returnDict = []
    for resultList in resultDict:
        clusterModel = resultList[1]
        clusterCount = resultList[2]
        nmiAll = resultList[3][0]
        accAll = resultList[3][1]
        nmiNoz = resultList[3][2]
        accNoz = resultList[3][3]
        emptyK = resultList[3][4]
        # , '*histCnt=', resultList[3][5][0:10]
        #baseResultsLab.append([str(clusterModel)+str(clusterCount)])
        #baseResultsVal.append(nmiNoz)
        #baseResults[str(clusterModel)+str(clusterCount)] = [nmiNoz]
        dataUsed = baseResultFileName.replace('.npy', '').replace('_baseResults', '')
        #print(valuesStrFormat2 %
        #(dataUsed, clusterModel, clusterCount, nmiAll, accAll, nmiNoz, accNoz, emptyK))
        returnDict.append([dataUsed, clusterModel, clusterCount, nmiAll, accAll, nmiNoz, accNoz, emptyK])

    df = pd.DataFrame(returnDict, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'accAll', 'nmiNoz', 'accNoz', 'emptyK'])
    funcH.setPandasDisplayOpts()
    if displayResults:
        print(df)

    return returnDict

def getBaseResults_Aug2020(dataToUse, pcaCount, numOfSigns, displayResults=True, baseResultFileName=''):
    if baseResultFileName == '':
        baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount,
                                               numOfSigns=numOfSigns, expectedFileType='BaseResultName')
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResultFileNameFull = os.path.join(baseLineResultFolder, baseResultFileName)
    resultDict = np.load(baseResultFileNameFull, allow_pickle=True)
    returnDict = []
    for resultList in resultDict:
        clusterModel = resultList[1]
        clusterCount = resultList[2]
        nmiAll = resultList[3][0]
        acc_cent = resultList[3][1]
        meanPurity_cent = resultList[3][2]
        weightedPurity_cent = resultList[3][3]
        acc_mxhs = resultList[3][4]
        meanPurity_mxhs = resultList[3][5]
        weightedPurity_mxhs = resultList[3][6]
        cnmxh_perc = resultList[3][7]
        emptyK = resultList[3][8]
        dataUsed = baseResultFileName.replace('.npy', '').replace('_baseResults', '')
        returnDict.append([dataUsed, clusterModel, clusterCount, nmiAll, acc_cent, meanPurity_cent, weightedPurity_cent, acc_mxhs, meanPurity_mxhs, weightedPurity_mxhs, cnmxh_perc, emptyK])

    df = pd.DataFrame(returnDict, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'acc_cent', 'meanPurity_cent', 'weightedPurity_cent', 'acc_mxhs', 'meanPurity_mxhs', 'weightedPurity_mxhs', 'cnmxh_perc', 'emptyK'])
    funcH.setPandasDisplayOpts()
    if displayResults:
        print(df)

    return returnDict

def traverseBaseResultsFolder():
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    fileList = funcH.getFileList(dir2Search=baseLineResultFolder, startString='', endString='.npy')
    brAll = []
    for f in fileList:
        br = getBaseResults(dataToUse='', normMode='', pcaCount=-1, numOfSigns=-1, displayResults=False, baseResultFileName=f)
        brAll = brAll + br
    returnDictAll = pd.DataFrame(brAll, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'accAll', 'nmiNoz', 'accNoz', 'emptyK']).sort_values(by='accNoz', ascending=False)
    print(returnDictAll)

    baseResults_csv = os.path.join(baseLineResultFolder, 'baseLineResults.csv')
    returnDictAll.to_csv(path_or_buf=baseResults_csv, sep=',', na_rep='NaN', float_format='%8.4f')

def traverseBaseResultsFolder_Aug2020():
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    fileList = funcH.getFileList(dir2Search=baseLineResultFolder, startString='', endString='.npy')
    brAll = []
    for f in fileList:
        br = getBaseResults_Aug2020(dataToUse='', pcaCount=-1, numOfSigns=-1, displayResults=False, baseResultFileName=f)
        brAll = brAll + br
    returnDictAll = pd.DataFrame(brAll, columns=['npyFileName', 'clusModel', 'clusCnt', 'nmiAll', 'acc_cent', 'meanPurity_cent', 'weightedPurity_cent', 'acc_mxhs', 'meanPurity_mxhs', 'weightedPurity_mxhs', 'cnmxh_perc', 'emptyK']).sort_values(by='cnmxh_perc', ascending=False)
    print(returnDictAll)

    baseResults_csv = os.path.join(baseLineResultFolder, 'baseLineResults.csv')
    returnDictAll.to_csv(path_or_buf=baseResults_csv, sep=',', na_rep='NaN', float_format='%8.4f')

def displayDataResults(method, dataToUse, normMode, pcaCount, numOfSigns, posteriorDim, weightReg = 1.0, batchSize = 16):
    getBaseResults(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, displayResults=True)
    modelParams = {
        "trainMode": method,
        "posterior_dim": posteriorDim,
        "weight_of_regularizer": weightReg,
        "dataToUse": dataToUse,
        "pcaCount": pcaCount,
        "numOfSigns": numOfSigns
    }
    trainParams = {
        "epochs": 0, "appendEpochBinary": 0, "applyCorr": 0, "corr_randMode": 0, "randomSeed": 5,
        "batch_size": batchSize
    }

    exp_name = createExperimentName(trainParams=trainParams, modelParams=modelParams, rnnParams={})
    csv_name, model_name, outdir = createExperimentDirectories(results_dir, exp_name)
    nmi_acc_file2load = outdir + os.sep + exp_name + '_nmi_acc.txt'

    epochIDs, nmiVals, accVals = np.loadtxt(nmi_acc_file2load, comments='#', delimiter="*", skiprows=1, unpack=True)

    _, lossVec, loss1Vec, mseVec = np.loadtxt(csv_name, comments='#', delimiter=";", skiprows=1, unpack=True)

    combinedResults = {
        'nmi': nmiVals,
        'acc': accVals,
        'loss': lossVec,
        'loss1': -100 * loss1Vec,
        'mse': mseVec
    }

    df = pd.DataFrame(combinedResults, index=[epochIDs])
    df.plot.line(title=exp_name)
    plt.pyplot.show()


    df = pd.DataFrame({'lab':baseResultsLab, 'val':baseResultsVal})
    ax = df.plot.bar(x='lab', y='val', rot=45)
    plt.pyplot.show()

def runForPred(labels_true, labels_pred, labelNames, predictDefStr):
    print("\r\n*-*-*start-", predictDefStr, "-end*-*-*\r\n")
    print("\r\n\r\n*-*-", predictDefStr, "calcClusterMetrics-*-*\r\n\r\n")
    klusRet = funcH.calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames)

    print("\r\n\r\n*-*-", predictDefStr, "calcCluster2ClassMetrics-*-*\r\n\r\n")
    classRet, _confMat, c_pdf, kr_pdf = funcH.calcCluster2ClassMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames, predictDefStr=predictDefStr)

    results_dir = funcH.getVariableByComputerName('results_dir')
    predictResultFold = os.path.join(results_dir, "predictionResults")
    funcH.createDirIfNotExist(predictResultFold)

    confMatFileName = predictDefStr + ".csv"
    confMatFileName = os.path.join(predictResultFold, confMatFileName)
    _confMat_df = pd.DataFrame(data=_confMat, index=labelNames, columns=labelNames)
    # _confMat_df = _confMat_df[(_confMat_df.T != 0).any()]
    pd.DataFrame.to_csv(_confMat_df, path_or_buf=confMatFileName)

    kr_pdf_FileName = "kluster_evaluations_" + predictDefStr + ".csv"
    kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)
    pd.DataFrame.to_csv(kr_pdf, path_or_buf=kr_pdf_FileName)

    c_pdf_FileName = "class_evaluations_" + predictDefStr + ".csv"
    c_pdf_FileName = os.path.join(predictResultFold, c_pdf_FileName)
    pd.DataFrame.to_csv(c_pdf, path_or_buf=c_pdf_FileName)

    print("*-*-*end-", predictDefStr, "-end*-*-*\r\n")
    return klusRet, classRet, _confMat, c_pdf, kr_pdf

def analayzePredictionResults(labels_pred, dataToUse, pcaCount, numOfSigns,
                              saveConfFigFileName='', predDefStr="predictionUnknown",
                              useNZ=True, confCalcMethod = 'dnn', confusionTreshold=0.3,
                              figMulCnt=None):
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName("results_dir")
    predictResultFold = os.path.join(results_dir, "predictionResults")

    fileName_detailedLabels = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=numOfSigns,expectedFileType='DetailedLabels')
    fileName_labels = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='Labels')
    detailed_labels_all = funcD.loadFileIfExist(data_dir, fileName_detailedLabels)
    labels_true = funcD.loadFileIfExist(data_dir, fileName_labels)
    labelNames = load_label_names(numOfSigns)

    labels_pred = labels_pred if not isinstance(labels_pred, str) else np.load(labels_pred)
    labels_true_nz, labels_pred_nz, _ = funcH.getNonZeroLabels(labels_true, labels_pred)

    if useNZ:
        labels_pred = labels_pred_nz
        labels_true = labels_true_nz-1
    else:
        labelNames.insert(0, "None")

    print(predDefStr)

    klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels_true, labels_pred, labelNames, predDefStr)

    print("\r\ncluster metrics comparison\r\n")
    print(klusRet)
    print("\r\n")

    print("\r\nclassification metrics comparison\r\n")
    print(classRet)
    print("\r\n")

    print("\r\nf1 score comparisons for classes\r\n")
    print(c_pdf)
    print("\r\n")

    klusRet.sort_index(inplace=True)
    kr_pdf_FileName = "kluster_evaluations_" + predDefStr + ".csv"
    kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)
    pd.DataFrame.to_csv(klusRet, path_or_buf=kr_pdf_FileName)

    print("confCalcMethod=", confCalcMethod)
    if confCalcMethod == 'dnn':
        acc, _confMat, kluster2Classes = funcH.getAccFromConf(labels_true, labels_pred)
    else:
        _confMat, kluster2Classes, kr_pdf, weightedPurity = funcH.countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=labelNames)

    if saveConfFigFileName != '':
        saveConfFigFileName = saveConfFigFileName.replace(".", "_ccm(" + confCalcMethod + ").")
        saveConfFigFileName = os.path.join(predictResultFold, saveConfFigFileName)

    #iterID = -1
    #normalizeByAxis = -1
    #add2XLabel = ''
    #add2YLabel = ''
    #funcH.plotConfMat(_confMat, labelNames, addCntXTicks=False, addCntYTicks=False, tickSize=10,
    #                  saveFileName=saveConfFigFileName, iterID=iterID,
    #                  normalizeByAxis=normalizeByAxis, add2XLabel=add2XLabel, add2YLabel=add2YLabel)
    fig, ax = funcH.plot_confusion_matrix(conf_mat=_confMat,
                                          colorbar=False,
                                          show_absolute=True,
                                          show_normed=True,
                                          class_names=labelNames,
                                          saveConfFigFileName=saveConfFigFileName,
                                          figMulCnt=figMulCnt,
                                          confusionTreshold=confusionTreshold)


def load_labels_pred_for_ensemble(useNZ=True, nos=11, featUsed="hgsk256",
                                  labels_preds_fold_name='/home/doga/Desktop/forBurak/wr1.0_hgsk256_11_bs16_cp2_cRM0_cSM1'):
    # load labels
    labelNames = load_label_names(nos)

    labels_filename = os.path.join(labels_preds_fold_name, "labels_" + str(nos) + ".npy")
    labels = np.load(labels_filename)
    labels_nz, _, _ = funcH.getNonZeroLabels(labels, labels)

    print("labels loaded of size", labels.shape)

    # load predictions as a list from a folder
    predFileList = funcH.getFileList(labels_preds_fold_name, startString="pd", endString=".npy", sortList=True)
    print(predFileList)

    predictionsDict = []
    N = predFileList.shape[0]
    for p in predFileList:
        predStr = featUsed + "_" + str(nos) + "_" + p.replace(".npy", "")

        preds_filename = os.path.join(labels_preds_fold_name, p)
        preds = np.load(preds_filename)
        print("preds(", predStr, ") loaded of size", preds.shape)
        if useNZ:
            _, preds, _ = funcH.getNonZeroLabels(labels, preds)
            print("preds = preds_nz of size", preds.shape)

        predictionsDict.append({"str": predStr, "prd": preds})

    if useNZ:
        labels = labels_nz - 1
        print("labels", labels.shape, " = labels_nz-1 because useNZ==True")
    else:
        labelNames.insert(0, "None")
        print("None is inserted at the beginning of labelNames because useNZ==False")

    print("labelNames = ")
    for x in range(len(labelNames)):
        print(x, ".", labelNames[x]),

    cluster_runs = None
    for i in range(0, N):
        cluster_runs = funcH.append_to_vstack(cluster_runs, predictionsDict[i]["prd"], dtype=int)

    return labelNames, labels, predictionsDict, cluster_runs, N


def load_labels_pred_for_ensemble_Aug2020(class_names, dataToUseVec=["hog", "sn", "sk"], nos=11, pcaCountVec=[96, 256],
                                          clusterModelsVec=["KMeans"], clustCntVec=[256]):
    results_dir = funcH.getVariableByComputerName('results_dir')

    predictionsDict = []
    N = 0
    for dataToUse in dataToUseVec:
        for pcaCount in pcaCountVec:
            baseResultFileName = funcD.getFileName(dataToUse=dataToUse, normMode="", pcaCount=pcaCount, numOfSigns=nos,
                                                   expectedFileType='BaseResultName')
            for clusterModel in clusterModelsVec:
                for curClustCnt in clustCntVec:
                    predictionFileName = baseResultFileName.replace("_baseResults.npy",
                                                                    "") + "_" + clusterModel + "_" + str(
                        curClustCnt) + ".npz"
                    predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
                    if os.path.isfile(predictionFileNameFull):
                        print("EXIST - ", predictionFileNameFull)

                        predStr = predictionFileName.replace(".npy", "").replace(".npz", "")

                        a = np.load(predictionFileNameFull)
                        labels_all = a["arr_0"]
                        predClusters = a["arr_1"]
                        _confMat_mapped_preds_mxhist, _, kr_pdf_mxhist, weightedPurity_mxhist, _ = funcH.countPredictionsForConfusionMat(
                            labels_all, predClusters, labelNames=class_names, centroid_info_pdf=None, verbose=0)
                        print("predStr", predStr)
                        print("preds(", predClusters, ") loaded of size", predClusters.shape)
                        predictionsDict.append({"str": predStr, "prd": predClusters})
                        N = N + 1

    cluster_runs = None
    for i in range(0, N):
        cluster_runs = funcH.append_to_vstack(cluster_runs, predictionsDict[i]["prd"], dtype=int)

    return class_names, labels_all, predictionsDict, cluster_runs, N

def ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                     consensus_clustering_max_k=256, useNZ=True, nos=11,
                     resultsToCombineDescriptorStr="",
                     labelNames = None, verbose=False):
    N = cluster_runs.shape[0]
    results_dir = funcH.getVariableByComputerName("results_dir")
    predictResultFold = os.path.join(results_dir, "predictionResults")

    # 1.run cluster_ensembles
    t = time.time()
    consensus_clustering_labels = funcEns.get_consensus_labels(cluster_runs,
                                                       consensus_clustering_max_k=consensus_clustering_max_k,
                                                       verbose=verbose)
    elapsed_cluster_ensembles = time.time() - t
    print('cluster_ensembles - elapsedTime({:4.2f})'.format(elapsed_cluster_ensembles), ', ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # 2.append ensembled consensus_clustering_labels into cluster_runs dictionary
    if resultsToCombineDescriptorStr == "":
        resultsToCombineDescriptorStr = "klusterResults_" + str(nos) + ("_nz_" if useNZ else "_") + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    resultsToCombineDescriptorStr = "klusterResults_" + str(nos) + ("_nz_" if useNZ else "_") + resultsToCombineDescriptorStr

    predictionsDict.append({"str": resultsToCombineDescriptorStr, "prd": consensus_clustering_labels})
    cluster_runs = funcH.append_to_vstack(cluster_runs, consensus_clustering_labels, dtype=int)

    # 3.for all clusterings run analysis of clusters and classes
    resultsDict = []
    for i in range(0, N + 1):
        t = time.time()
        klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels, predictionsDict[i]["prd"], labelNames, predictionsDict[i]["str"])
        elapsed_runForPred = time.time() - t
        print('runForPred(', predictionsDict[i]["str"], ') - elapsedTime({:4.2f})'.format(elapsed_runForPred),
              ' ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        resultsDict.append({"klusRet": klusRet, "classRet": classRet,
                            "_confMat": _confMat, "c_pdf": c_pdf, "kr_pdf": kr_pdf})

    klusRet = resultsDict[0]["klusRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        klusRet.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["klusRet"]['value'], True)
    print("\r\ncluster metrics comparison\r\n")
    print(klusRet)
    print("\r\n")

    classRet = resultsDict[0]["classRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        classRet.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["classRet"]['value'], True)
    if verbose:
        print("\r\nclassification metrics comparison\r\n")
        print(classRet)
        print("\r\n")
    class_metrics_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "class_metrics") + ".csv")
    pd.DataFrame.to_csv(classRet, path_or_buf=class_metrics_FileName)

    c_pdf = resultsDict[0]["c_pdf"][['class', '%f1']].sort_index().rename(
        columns={"class": "f1Score", "%f1": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        c_pdf.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["c_pdf"][['%f1']].sort_index(), True)
    if verbose:
        print("\r\nf1 score comparisons for classes\r\n")
        print(c_pdf)
        print("\r\n")
    f1_comparison_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "f1_comparison") + ".csv")
    pd.DataFrame.to_csv(c_pdf, path_or_buf=f1_comparison_FileName)

    print('calc_ensemble_driven_cluster_index - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    eci_vec, clusterCounts = funcEns.calc_ensemble_driven_cluster_index(cluster_runs=cluster_runs)
    elapsed = time.time() - t
    print('calc_ensemble_driven_cluster_index - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_LWCA_matrix - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    lwca_mat = funcEns.create_LWCA_matrix(cluster_runs, eci_vec=eci_vec, verbose=0)
    elapsed = time.time() - t
    print('create_LWCA_matrix - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_quality_vec - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    quality_vec = funcEns.calc_quality_weight_basic_clustering(cluster_runs, logType=0, verbose=0)
    elapsed = time.time() - t
    print('create_quality_vec - elapsedTime({:4.2f})'.format(elapsed), ' ended at ',
          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sampleCntToPick = np.array([1, 3, 5, 10], dtype=int)
    columns = ['1', '3', '5', '10']
    columns_2 = ['c1', 'c3', 'c5', 'c10']
    colCnt = len(columns)

    resultTable_mat = np.zeros((colCnt+2,N+1))
    resultTable_pd_columns = ['c1', 'c3', 'c5', 'c10', 'All', 'sum']
    resultTable_pd_index = []
    resultTable_FileName = os.path.join(predictResultFold, resultsToCombineDescriptorStr.replace("klusterResults", "resultTable") + ".csv")

    #cluster_runs_cmbn = []
    for i in range(0, N+1):
        kr_pdf_cur = resultsDict[i]["kr_pdf"]
        eci_vec_cur = eci_vec[i].copy()
        predictDefStr = predictionsDict[i]["str"]
        resultTable_pd_index.append(predictDefStr)
        #cluster_runs_cmbn = funcH.append_to_vstack(cluster_runs_cmbn, predictionsDict[i]["prd"], dtype=int)
        print(predictDefStr, "Quality of cluster = {:6.4f}".format(quality_vec[i]), "number of clusters : ", kr_pdf_cur.shape)
        predictions_cur = predictionsDict[i]["prd"]
        unique_preds = np.unique(predictions_cur)

        kr_pdf_cur.sort_index(inplace=True)
        eci_N = np.array(eci_vec_cur * kr_pdf_cur['N'], dtype=float)
        eci_pd = pd.DataFrame(eci_vec_cur, columns=['ECi'])
        eci_N_pd = pd.DataFrame(eci_N, columns=['ECi_n'])
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd], axis=1)
        pd_comb.sort_values(by=['ECi_n', 'N'], inplace=True, ascending=[False, False])

        kr_pdf_FileName = "kluster_evaluations_" + predictDefStr + ".csv"
        kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)

        cols2add = np.zeros((clusterCounts[i], colCnt), dtype=float)
        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        cols2add_2 = np.zeros((clusterCounts[i], colCnt), dtype=float)
        cols2add_2_pd = pd.DataFrame(cols2add_2, columns=columns_2)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd, cols2add_2_pd], axis=1)

        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

        # pick first 10 15 20 25 samples according to lwca_mat
        for pi in range(0, clusterCounts[i]):
            cur_pred = unique_preds[pi]
            predictedSamples = funcH.getInds(predictions_cur, cur_pred)
            sampleLabels = labels[predictedSamples]
            lwca_cur = lwca_mat[predictedSamples, :]
            lwca_cur = lwca_cur[:, predictedSamples]
            simSum = np.sum(lwca_cur, axis=0) + np.sum(lwca_cur, axis=1).T
            v, idx = funcH.sortVec(simSum)
            sortedPredictionsIdx = predictedSamples[idx]
            sortedLabelIdx = labels[sortedPredictionsIdx]
            curSampleCntInCluster = len(sampleLabels)
            mappedClassOfKluster = funcH.get_most_frequent(list(sortedLabelIdx))

            for sj in range(0, colCnt):
                sCnt = sampleCntToPick[sj] if curSampleCntInCluster>sampleCntToPick[sj] else curSampleCntInCluster
                sampleLabelsPicked = sortedLabelIdx[:sCnt]
                purity_k, correctLabelInds, mappedClass, _ = funcH.calcPurity(list(sampleLabelsPicked))
                if mappedClass == mappedClassOfKluster:
                    cols2add[pi, sj] = purity_k
                else:
                    cols2add[pi, sj] = -mappedClass+(mappedClassOfKluster/100)
                cols2add_2[pi, sj] = np.sum(sortedLabelIdx == mappedClass)

        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        cols2add_2_pd = pd.DataFrame(cols2add_2, columns=columns_2)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd, cols2add_2_pd], axis=1)
        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

        allPredCorrectSum = np.asarray(np.sum(kr_pdf_cur.iloc[:,2:3]))
        numOfSamples = np.asarray(np.sum(kr_pdf_cur.iloc[:, 3:4]))
        precidtCols = np.sum(cols2add_2, axis=0, keepdims=True)
        resultTable_mat[0:colCnt, i] = precidtCols.T.squeeze()
        resultTable_mat[colCnt, i] = allPredCorrectSum
        resultTable_mat[colCnt+1, i] = numOfSamples

    resultTable_mat[:-1, :] = resultTable_mat[:-1, :] / resultTable_mat[-1, :]
    resultTable_pd = pd.DataFrame(resultTable_mat, index=resultTable_pd_columns, columns=resultTable_pd_index)
    pd.DataFrame.to_csv(resultTable_pd, path_or_buf=resultTable_FileName)

    _confMat_consensus, _, kr_pdf_consensus, weightedPurity_consensus, _ = funcH.countPredictionsForConfusionMat(labels, consensus_clustering_labels, labelNames=labelNames, centroid_info_pdf=None, verbose=0)
    meanPurity_consensus = np.mean(np.asarray(kr_pdf_consensus["%purity"]))
    nmi_consensus = funcH.get_nmi_only(labels, consensus_clustering_labels)
    acc_consensus = np.sum(np.diag(_confMat_consensus)) / np.sum(np.sum(_confMat_consensus))
    print(resultsToCombineDescriptorStr, "\nnmi_consensus", nmi_consensus, "acc_consensus", acc_consensus, "meanPurity_consensus", meanPurity_consensus)
    del(_confMat_consensus, kr_pdf_consensus, weightedPurity_consensus)

def plot_supervised_results(fold_name, model_name="resnet18", random_seed=1, nos=11):
    result_mat = np.zeros((6, 5), dtype=float)
    # fold_name = "/home/doga/DataFolder/sup_old/results_old_to_check"
    # fold_name = "/home/doga/DataFolder/sup/results"
    acc_list = {"ep": [], "tr": [], "va": [], "te": [], }
    usr_list = {"u2": [], "u3": [], "u5": [], "u6": [], "u7": [], }
    for i, userIDTest in enumerate({2, 3, 4, 5, 6, 7}):
        user_id_str = "u"+str(userIDTest)
        usr_list[user_id_str] = {"ep": [], "tr": [], "va": [], "te": [], }
        for j, crossValidID in enumerate({1, 2, 3, 4, 5}):  # 32
            file_name = "rCF_te" + str(userIDTest) + "_cv" + str(crossValidID) + \
                        "_" + model_name + \
                        "_" + "neuralNetHandImages_nos" + str(nos) + "_rs224" + \
                        "_rs" + str(random_seed).zfill(2) + ".csv"
            file2read = os.path.join(fold_name, file_name)
            try:
                featsMat = pd.read_csv(file2read, header=0, sep="*", names=["epoch", "train", "validation", "test"])
                max_val = np.max(featsMat["test"].values[:])
                acc_list["ep"].append(featsMat["epoch"].values[:])
                acc_list["tr"].append(featsMat["train"].values[:])
                acc_list["va"].append(featsMat["validation"].values[:])
                acc_list["te"].append(featsMat["test"].values[:])

                usr_list[user_id_str]["ep"].append(featsMat["epoch"].values[1:])
                usr_list[user_id_str]["tr"].append(featsMat["train"].values[1:])
                usr_list[user_id_str]["va"].append(featsMat["validation"].values[1:])
                usr_list[user_id_str]["te"].append(featsMat["test"].values[1:])

            except Exception as e:
                print(str(e))
                max_val = np.nan
            result_mat[i, j] = max_val
    result_pd = pd.DataFrame(result_mat, columns=["cv1", "cv2", "cv3", "cv4", "cv5"],
                             index=["u2", "u3", "u4", "u5", "u6", "u7"])
    print(result_pd)
    file_name = "rCF_all" \
                "_" + model_name + \
                "_" + "neuralNetHandImages_nos" + str(nos) + "_rs224" + \
                "_rs" + str(random_seed).zfill(2) + ".csv"
    fileNameFull_csv = os.path.join(fold_name, file_name)
    print(fileNameFull_csv)
    result_pd.to_csv(fileNameFull_csv, index=["u2", "u3", "u4", "u5", "u6", "u7"], header=True)

    data_set_ident_str = "HospiSign Development Dataset" if nos==11 else "HospiSign Expanded Dataset"
    dev_exp_str = "development" if nos==11 else "expanded"

    save_fig_name = model_name + "_" + dev_exp_str + "_accRange_per_user.png"
    title_str = "{:s}{:s} Model({:s}){:s} 5-Fold-Cross-Validation Accuracy Range".format(data_set_ident_str, os.linesep, model_name, os.linesep)
    funcVis.stack_fig_disp(result_mat, fold_name, save_fig_name, title_str)

    mmm_mat = np.column_stack((np.nanmin(result_mat[:, :-1], axis=1).ravel(),
                               np.nanmean(result_mat[:, :-1], axis=1).ravel(),
                               np.nanmax(result_mat[:, :-1], axis=1).ravel()))
    result_mmm = pd.DataFrame(mmm_mat, index=["u2", "u3", "u4", "u5", "u6", "u7"], columns=["min", "mean", "max"])
    print(result_mmm)

    save_fig_name = model_name + "_accBar_per_user.png"
    title_str = model_name + " Accuracy Bar-Plot for 5-Fold-Cross-Validation"
    funcVis.pdf_bar_plot_users(result_mmm, fold_name, save_fig_name, title_str)

    fig = funcVis.plot_acc_eval(acc_list, "te", model_name + " Test Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_te_all_range.png"), bbox_inches='tight')

    fig = funcVis.plot_acc_eval(acc_list, "tr", model_name + " Train Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_tr_all_range.png"), bbox_inches='tight')

    fig = funcVis.plot_acc_eval(acc_list, "va", model_name + " Validation Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_va_all_range.png"), bbox_inches='tight')

    for i, userIDTest in enumerate({2, 3, 4, 5, 6, 7}):
        title_str = data_set_ident_str + os.linesep + "Model({:s}), User({:d}){:s}5-Fold-Cross-Validation Accuracy Range".format(model_name, userIDTest, os.linesep)
        filename_2_save = "{:s}_u{:d}_{:s}_acc_range.png".format(model_name, userIDTest, dev_exp_str)
        fig = funcVis.plot_acc_range_for_user(usr_list, userIDTest, title_str)
        fig.savefig(os.path.join(fold_name, filename_2_save), bbox_inches='tight')

    plt.pyplot.close('all')

def plot_supervised_results_aug2020(fold_name, model_name="resnet18", random_seed=1, nos=11):
    result_mat = np.zeros((6, 6), dtype=float)
    # fold_name = "/home/doga/DataFolder/sup_old/results_old_to_check"
    # fold_name = "/home/doga/DataFolder/sup/results"
    acc_list = {"ep": [], "tr": [], "va": [], "te": [], }
    usr_list = {"u2": [], "u3": [], "u5": [], "u6": [], "u7": [], }
    for userIDTest in [2, 3, 4, 5, 6, 7]:
        user_id_str = "u"+str(userIDTest)
        usr_list[user_id_str] = {"ep": [], "tr": [], "va": [], "te": [], }
        for userValidID in [2, 3, 4, 5, 6, 7]:  # 32
            result_mat[userIDTest - 2, userValidID - 2] = np.nan
            if userValidID == userIDTest:
                continue
            file_name = "rCF_te" + str(userIDTest) + "_va" + str(userValidID) + \
                        "_" + model_name + \
                        "_" + "neuralNetHandImages_nos" + str(nos) + "_rs224" + \
                        "_rs" + str(random_seed).zfill(2) + ".csv"
            file2read = os.path.join(fold_name, file_name)
            try:
                featsMat = pd.read_csv(file2read, header=0, sep="*", names=["epoch", "train", "validation", "test"])
                accvecva = featsMat["validation"].values[:]
                accvecte = featsMat["test"].values[:]

                acc_list["ep"].append(featsMat["epoch"].values[:])
                acc_list["tr"].append(featsMat["train"].values[:])
                acc_list["va"].append(accvecva)
                acc_list["te"].append(accvecte)

                bestVaID = np.argmax(accvecva)
                bestTeID = np.argmax(accvecte)
                formatStr = "5.3f"
                print(("bestVaID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(
                    bestVaID, accvecva[bestVaID], accvecte[bestVaID]))
                print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(
                    bestTeID, accvecva[bestTeID], accvecte[bestTeID]))
                print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1],
                                                                                               accvecte[-1]))
                max_val = accvecte[bestVaID]

                usr_list[user_id_str]["ep"].append(featsMat["epoch"].values[1:])
                usr_list[user_id_str]["tr"].append(featsMat["train"].values[1:])
                usr_list[user_id_str]["va"].append(featsMat["validation"].values[1:])
                usr_list[user_id_str]["te"].append(featsMat["test"].values[1:])

            except Exception as e:
                print(str(e))
                max_val = np.nan
            result_mat[userIDTest-2, userValidID-2] = max_val
    result_pd = pd.DataFrame(result_mat, columns=["u2", "u3", "u4", "u5", "u6", "u7"],
                             index=["u2", "u3", "u4", "u5", "u6", "u7"])
    print(result_pd)
    file_name = "rCF_all" \
                "_" + model_name + \
                "_" + "neuralNetHandImages_nos" + str(nos) + "_rs224" + \
                "_rs" + str(random_seed).zfill(2) + ".csv"
    fileNameFull_csv = os.path.join(fold_name, file_name)
    print(fileNameFull_csv)
    result_pd.to_csv(fileNameFull_csv, index=["u2", "u3", "u4", "u5", "u6", "u7"], header=True, quoting=csv.QUOTE_NONE)

    data_set_ident_str = "HospiSign Development Dataset" if nos==11 else "HospiSign Expanded Dataset"
    dev_exp_str = "development" if nos==11 else "expanded"

    save_fig_name = model_name + "_" + dev_exp_str + "_accRange_per_user.png"
    title_str = "{:s}{:s} Model({:s}){:s} 5-Fold-Cross-Validation Accuracy Range".format(data_set_ident_str, os.linesep, model_name, os.linesep)
    funcVis.stack_fig_disp(result_mat, fold_name, save_fig_name, title_str)

    mmm_mat = np.column_stack((np.nanmin(result_mat[:, :], axis=1).ravel(),
                               np.nanmean(result_mat[:, :], axis=1).ravel(),
                               np.nanmax(result_mat[:, :], axis=1).ravel()))
    result_mmm = pd.DataFrame(mmm_mat, index=["u2", "u3", "u4", "u5", "u6", "u7"], columns=["min", "mean", "max"])
    print(result_mmm)

    save_fig_name = model_name + "_accBar_per_user.png"
    title_str = model_name + " Accuracy Bar-Plot for 5-Fold-Cross-Validation"
    funcVis.pdf_bar_plot_users(result_mmm, fold_name, save_fig_name, title_str)

    fig = funcVis.plot_acc_eval(acc_list, "te", model_name + " Test Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_te_all_range.png"), bbox_inches='tight')

    fig = funcVis.plot_acc_eval(acc_list, "tr", model_name + " Train Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_tr_all_range.png"), bbox_inches='tight')

    fig = funcVis.plot_acc_eval(acc_list, "va", model_name + " Validation Accuracy Range for All Users 5-Fold-Cross-Validation")
    fig.savefig(os.path.join(fold_name, model_name + "_va_all_range.png"), bbox_inches='tight')

    for i, userIDTest in enumerate({2, 3, 4, 5, 6, 7}):
        title_str = data_set_ident_str + os.linesep + "Model({:s}), User({:d}){:s}5-Fold-Cross-Validation Accuracy Range".format(model_name, userIDTest, os.linesep)
        filename_2_save = "{:s}_u{:d}_{:s}_acc_range.png".format(model_name, userIDTest, dev_exp_str)
        fig = funcVis.plot_acc_range_for_user(usr_list, userIDTest, title_str)
        fig.savefig(os.path.join(fold_name, filename_2_save), bbox_inches='tight')

    plt.pyplot.close('all')


def conf_mat_update(old_conf, new_conf, op_name="best_cell"):
    if len(old_conf)==0:
        updated_conf = new_conf
    elif op_name=="best_cell":
        #  max val stays at diagonal
        #  min val stays at other places
        updated_conf = np.minimum(old_conf, new_conf)
        diag_to_set = np.max([np.diag(old_conf), np.diag(new_conf)], axis=0)
        np.fill_diagonal(updated_conf, diag_to_set)
    elif op_name=="worst_cell":
        #  max val stays at diagonal
        #  min val stays at other places
        updated_conf = np.maximum(old_conf, new_conf)
        diag_to_set = np.min([np.diag(old_conf), np.diag(new_conf)], axis=0)
        np.fill_diagonal(updated_conf, diag_to_set)
    return updated_conf

def sup_res_confusion_run(fold_name, model_name="resnet18", random_seed=1, nos=11):
    # fold_name = "/home/doga/DataFolder/sup/preds_resnet18"
    for i, userIDTest in enumerate({2, 3, 4, 5, 6, 7}):
        conf_size = []
        conf_std = {"tr": [], "va": [], "te": [], }
        conf_best = {"tr": [], "va": [], "te": [], }
        conf_worst = {"tr": [], "va": [], "te": [], }
        lab_cnts = {"tr": [], "va": [], "te": [], }
        for j, crossValidID in enumerate({1, 2, 3, 4, 5}):  # 32
            fold_name_uc = "pred_te" + str(userIDTest) + "_cv" + str(crossValidID) + \
                        "_" + model_name + \
                        "_" + "neuralNetHandImages_nos" + str(nos) + "_rs224" + \
                        "_rs" + str(random_seed).zfill(2)
            fold_name_uc = os.path.join(fold_name, fold_name_uc)
            if not os.path.isdir(fold_name_uc):
                continue
            ep_file_list = funcH.getFileList(dir2Search=fold_name_uc, endString=".npy", startString="ep", sortList=True)
            ep_cnt = len(ep_file_list)-1  # remove ep-1
            conf_list = {"tr": [], "va": [], "te": [], }
            acc_list = {"tr": [], "va": [], "te": [], }
            for ep in range(0, ep_cnt):
                npy_file = os.path.join(fold_name_uc, "ep{:03d}.npy".format(ep))
                results_dict = np.load(npy_file, allow_pickle=True)

                tra_lab = results_dict.item().get('labels_tra')
                tra_prd = results_dict.item().get('pred_tra')
                _confMat_tra = confusion_matrix(tra_lab, tra_prd)
                acc_tra = np.sum(np.diag(_confMat_tra)) / np.sum(np.sum(_confMat_tra))

                val_lab = results_dict.item().get('labels_val')
                val_prd = results_dict.item().get('pred_val')
                _confMat_val = confusion_matrix(val_lab, val_prd)
                acc_val = np.sum(np.diag(_confMat_val)) / np.sum(np.sum(_confMat_val))

                tes_lab = results_dict.item().get('labels_tes')
                tes_prd = results_dict.item().get('pred_tes')
                _confMat_tes = confusion_matrix(tes_lab, tes_prd)
                acc_tes = np.sum(np.diag(_confMat_tes)) / np.sum(np.sum(_confMat_tes))

                conf_size = _confMat_tes.shape

                acc_list["tr"].append(acc_tra)
                acc_list["va"].append(acc_val)
                acc_list["te"].append(acc_tes)

                conf_list["tr"].append(_confMat_tra)
                conf_list["va"].append(_confMat_val)
                conf_list["te"].append(_confMat_tes)

                lab_cnts["tr"] = np.sum(_confMat_tra, axis=1)
                lab_cnts["va"] = np.sum(_confMat_val, axis=1)
                lab_cnts["te"] = np.sum(_confMat_tes, axis=1)

            conf_std["tr"].append(_confMat_tra.reshape(1, -1).squeeze())
            conf_std["va"].append(_confMat_val.reshape(1, -1).squeeze())
            conf_std["te"].append(_confMat_tes.reshape(1, -1).squeeze())

            conf_best["tr"] = conf_mat_update(conf_best["tr"], _confMat_tra, op_name="best_cell")
            conf_worst["tr"] = conf_mat_update(conf_worst["tr"], _confMat_tra, op_name="worst_cell")
            conf_best["va"] = conf_mat_update(conf_best["va"], _confMat_val, op_name="best_cell")
            conf_worst["va"] = conf_mat_update(conf_worst["va"], _confMat_val, op_name="worst_cell")
            conf_best["te"] = conf_mat_update(conf_best["te"], _confMat_tes, op_name="best_cell")
            conf_worst["te"] = conf_mat_update(conf_worst["te"], _confMat_tes, op_name="worst_cell")

            # plot conf_worst, conf_std, conf_best for each tr/va/te as sub plots

def get_label_names_resnet(te_u_id=6, cv_id=1, nos=11, ep=59, model="resnet18",
                           pred_fold_base="/home/doga/DataFolder/sup"):
    pred_fold_cur = "preds_" + model + os.path.sep + "pred_te" + str(te_u_id) + "_cv" + str(
        cv_id) + "_" + model + "_neuralNetHandImages_nos" + str(nos) + "_rs224_rs01"
    pred_file = "ep{:03d}.npy".format(ep)
    predFileName = os.path.join(pred_fold_base, pred_fold_cur, pred_file)

    X = np.load(predFileName, allow_pickle=True)
    la_tr, pr_tr = X.item().get('labels_tra'), X.item().get('pred_tra')
    la_va, pr_va = X.item().get('labels_val'), X.item().get('pred_val')
    la_te, pr_te = X.item().get('labels_tes'), X.item().get('pred_tes')

    csv_fold = os.path.join(pred_fold_base, "data",
                            "data_te" + str(te_u_id) + "_cv" + str(cv_id) + "_neuralNetHandImages_nos" + str(
                                nos) + "_rs224_rs01")
    csv_fname = "cnt_table_te" + str(te_u_id) + "_cv" + str(cv_id) + "_resnet18neuralNetHandImages_nos" + str(
        nos) + "_rs224.csv"
    csv_fall = os.path.join(csv_fold, csv_fname)
    cnt_csv = pd.read_csv(csv_fall, header=0, names=["khsnames", "train", "validation", "test", "total"])
    labelNames = cnt_csv["khsnames"].values[:-1].copy()
    khs_cnt_vec_csv_tr = cnt_csv["train"].values[:-1].copy()
    khs_cnt_vec_csv_va = cnt_csv["validation"].values[:-1].copy()
    khs_cnt_vec_csv_te = cnt_csv["test"].values[:-1].copy()

    conf_mat_tr = confusion_matrix(la_tr, pr_tr)
    khs_cnt_vec_conf_tr = np.sum(conf_mat_tr, axis=1)
    conf_mat_te = confusion_matrix(la_te, pr_te)
    khs_cnt_vec_conf_te = np.sum(conf_mat_te, axis=1)
    conf_mat_va = confusion_matrix(la_va, pr_va)
    khs_cnt_vec_conf_va = np.sum(conf_mat_va, axis=1)

    i = 0
    khsNamesMatchInds = np.zeros(khs_cnt_vec_conf_tr.shape, dtype=int)
    for i in range(0, np.size(khs_cnt_vec_conf_tr)):
        cnt_co_tr = khs_cnt_vec_conf_tr[i]
        cnt_co_te = khs_cnt_vec_conf_te[i]
        cnt_co_va = khs_cnt_vec_conf_va[i]
        # these counts should match the csv counts
        result = np.where(
            ((khs_cnt_vec_csv_tr == cnt_co_tr) & (khs_cnt_vec_csv_va == cnt_co_va) & (khs_cnt_vec_csv_te == cnt_co_te)))
        #print(labelNames[result[0][0]], result[0], cnt_co_tr, cnt_co_va, cnt_co_te, np.size(result[0]))
        khsNamesMatchInds[i] = result[0][0]
        khs_cnt_vec_csv_tr[result[0][0]] = -1
        khs_cnt_vec_csv_va[result[0][0]] = -1
        khs_cnt_vec_csv_te[result[0][0]] = -1
        i = i + 1
    labelNamesNew = labelNames[khsNamesMatchInds]
    conf_mat = {"tr": conf_mat_tr,
                "te": conf_mat_te,
                "va": conf_mat_va, }
    return labelNamesNew, conf_mat

def prepare_data_4(X, y, detailedLabels, validUser, testUser):
    print("y.dtype=", type(y))
    y = np.asarray(y, dtype=int)
    uniqLabels = np.unique(y)
    #print("uniqLabels=", uniqLabels)
    #print("uniqLabels2=", np.unique(detailedLabels[:, 2]))

    tr_inds = np.array([], dtype=int)
    va_inds = np.array([], dtype=int)
    te_inds = np.array([], dtype=int)
    empty_test_classes = {}
    empty_validation_classes = {}
    for label in uniqLabels:
        inds = funcH.getInds(y, label)
        cnt = len(inds)

        #print("label=", label, ", inds=", inds, ", cnt=", cnt)
        #print("detailedLabels=", detailedLabels[inds, :])
        if cnt > 0:
            # userid = testUser to test
            # userid = validUser to valid
            # others to train
            userIDs = detailedLabels[inds, 1]
            #print("unique userIDs = ", np.unique(userIDs))
            teIDs = funcH.getInds(userIDs, testUser)
            if teIDs.shape[0] > 0:
                teIndsCur = inds[teIDs]
                te_inds = np.concatenate((te_inds, teIndsCur))
            else:
                teIndsCur = []
                empty_test_classes[label] = 1

            vaIDs = funcH.getInds(userIDs, validUser)
            if vaIDs.shape[0] > 0:
                vaIndsCur = inds[vaIDs]
                va_inds = np.concatenate((va_inds, vaIndsCur))
            else:
                vaIndsCur = []
                empty_validation_classes[label] = 1

            # print("label=",label,", teIDs=", teIDs.shape)
            # print("label=",label,", vaIDs=", vaIDs.shape)

            usedInds = np.asarray(np.squeeze(np.concatenate((teIDs, vaIDs))), dtype=int)
            # print("label=",label,", usedInds=", usedInds.shape)

            allInds = np.arange(0, len(inds))
            trIDs = np.delete(allInds, usedInds)
            trIndsCur = inds[trIDs]
            tr_inds = np.concatenate((tr_inds, trIndsCur))

            trCnt = len(trIndsCur)
            vaCnt = len(vaIndsCur)
            teCnt = len(teIndsCur)

            if cnt != trCnt + vaCnt + teCnt:
                print("xxxxxx cnt all = ", cnt, "!=", trCnt + vaCnt + teCnt)
            print("label(", label, "),cnt(", cnt, "),trCnt(", trCnt, "),vaCnt(", vaCnt, "),teCnt(", teCnt, ")")
    print("len train = ", len(tr_inds))
    print("len valid = ", len(va_inds))
    print("len test = ", len(te_inds))
    print("len all = ", len(y), "=", len(tr_inds) + len(va_inds) + len(te_inds))

    print("empty_test_classes", empty_test_classes)
    print("empty_validation_classes", empty_validation_classes)

    dataset_tr = funcD.HandCraftedDataset("", X=X[tr_inds, :], y=y[tr_inds])
    train_dl = DataLoader(dataset_tr, batch_size=32, shuffle=True)
    dataset_va = funcD.HandCraftedDataset("", X=X[va_inds, :], y=y[va_inds])
    valid_dl = DataLoader(dataset_va, batch_size=1024, shuffle=False)
    dataset_te = funcD.HandCraftedDataset("", X=X[te_inds, :], y=y[te_inds])
    test_dl = DataLoader(dataset_te, batch_size=1024, shuffle=False)

    return train_dl, valid_dl, test_dl

def get_hospisign_labels(nos=11, sortBy=None, verbose=0):
    base_dir = funcH.getVariableByComputerName('base_dir')
    baseFold = os.path.join(base_dir, "neuralNetHandImages_nos" + str(nos) + "_rs224", "imgs")
    list_dict_file = os.path.join(baseFold, "list_dict.txt")
    if not os.path.isfile(list_dict_file):
        savefilename = "list_dict.txt"
        web_filename = "list_dict_" + ("Dev" if nos == 11 else "Exp") + "Set.txt"
        print(web_filename, "-->", savefilename)
        download_hospisign_data(baseFold, web_filename, savefilename)

    a = pd.read_csv(list_dict_file, delimiter="*", header=None,
                    names=["sign", "user", "rep", "frameID", "khsID", "khsName", "hand"])
    b, uniqKHSinds = np.unique(np.asarray(a["khsID"]), return_index=True)
    labelsAll = np.asarray(a["khsID"], dtype=int)
    namesAll = np.asarray(a["khsName"])

    print(len(np.unique(namesAll)))
    print(np.unique(namesAll))

    labels_sui = np.squeeze(np.asarray(a[["sign", "user", "khsID"]]))
    # get sort index first
    assignedKHSinds = labelsAll[uniqKHSinds]
    #selectedKHSnames = np.array([str(np.char.strip(n)) for n in namesAll[uniqKHSinds]])
    selectedKHSnames = []
    for cur_ind in uniqKHSinds:
        cur_label = labelsAll[cur_ind]
        label_samples = funcH.getInds(labelsAll, cur_label)
        label_names = namesAll[label_samples]
        cur_names = np.asarray([np.char.strip(nm) for nm in np.unique(label_names)])
        cur_name = '|'.join(cur_names)
        selectedKHSnames.append(cur_name)

    selectedKHSnames = np.asarray(selectedKHSnames)
    if verbose > 1:
        print(labelsAll.shape, labelsAll.dtype)
    sortedLabelsAll, sortedLabelsMap = funcH.reset_labels(labelsAll, assignedKHSinds, selectedKHSnames, sortBy=sortBy,
                                                          verbose=verbose)
    if verbose > 1:
        print("sortedLabelsAll:\n", sortedLabelsAll.head())
        print("sortedLabelsMap:\n", sortedLabelsMap)
        print(labels_sui.shape, labels_sui.dtype)
    labels_sui[:, 2] = np.squeeze(np.array(sortedLabelsAll))

    lb_map = np.vstack((sortedLabelsMap["labelIDs"], sortedLabelsMap["labelStrings"])).T

    x = Counter(np.squeeze(labelsAll).astype(int))

    khsCntVec = [v for k, v in x.most_common()]
    khsIndex = [k for k, v in x.most_common()]
    if verbose > 2:
        print("x:\n", x)
        khsNameCol = [str(np.squeeze(lb_map[np.where(lb_map[:, 0] == k), 1])) for k, v in x.most_common()]
        print("khsNameCol:\n", khsNameCol)
        print("khsCntVec:\n", khsCntVec)
        print("khsIndex:\n", khsIndex)
    khsCntCol = np.asarray(khsCntVec)[np.argsort(khsIndex)]
    if verbose > 2:
        print("khsCntVec(sorted accordingly):\n", khsCntCol)

    lb_map_new = pd.DataFrame({"khsID": lb_map[:, 0], "khsName": lb_map[:, 1], "khsCnt": khsCntCol})
    lb_map_cnt = lb_map_new.sort_values(by='khsCnt', ascending=False)
    lb_map_id = lb_map_new.sort_values(by='khsID', ascending=True)
    lb_map_name = lb_map_new.sort_values(by='khsName', ascending=True)
    if verbose > 1:
        print("lb_map_cnt=\n", lb_map_cnt)
        print("lb_map_id=\n", lb_map_id)
        print("lb_map_name=\n", lb_map_name)

    hospisign_labels = {
        "labels": sortedLabelsAll,
        "labels_sui": labels_sui,
        "khsInds": sortedLabelsMap["labelIDs"],
        "khsNames": sortedLabelsMap["labelStrings"],
        "label_map": lb_map_id,
        "label_map_cnt": lb_map_cnt,
        "label_map_name": lb_map_name,
    }

    return hospisign_labels

def download_hospisign_data(baseFold, web_filename, savefilename):
    mat = os.path.join(baseFold, savefilename)
    if not os.path.isfile(mat):
        url = "ftp://dogasiyli:Doga.Siyli@dogasiyli.com/hospisign.dogasiyli.com/extractedData/" + web_filename
        funcH.download_file(link_adr=url, save2path=baseFold, savefilename=savefilename)
        web_to_disk_fname = os.path.join(baseFold, web_filename)
        if os.path.isfile(web_to_disk_fname) and not os.path.isfile(mat):
            os.rename(web_to_disk_fname, mat)
    return

def get_hospisign_feats(nos=11, labelsSortBy=None, verbose=0):
    base_dir = funcH.getVariableByComputerName('base_dir')
    baseFold = os.path.join(base_dir, "neuralNetHandImages_nos" + str(nos) + "_rs224", "imgs")

    hogMat = os.path.join(baseFold, "hog_10_9.mat")
    skelMat = os.path.join(baseFold, "skel.mat")
    snMat = os.path.join(baseFold, "sn.mat")

    funcH.createDirIfNotExist(baseFold)
    #download data from mendeleyData to reuired place
    for f in ['hog', 'skel', 'sn']:
        savefilename = f.replace("hog", "hog_10_9") + ".mat"
        web_filename = f.replace("hog", "hog_").replace("skel", "skeleton_").replace("sn", "snv_") + ("Dev" if nos==11 else "Exp") + "Set.mat"
        print(web_filename, "-->", savefilename)
        download_hospisign_data(baseFold, web_filename, savefilename)

    hg_ft = funcH.loadMatFile(hogMat, verbose=verbose)
    sn_ft = funcH.loadMatFile(snMat, verbose=verbose)
    sk_ft = funcH.loadMatFile(skelMat, verbose=verbose)

    hg_ft = hg_ft['hogImArr']
    sn_ft = sn_ft['mat2sa']
    sk_ft = sk_ft['mat2sa']

    if verbose > 0:
        print("hog = ", hg_ft.shape, hg_ft.dtype)
        print("surfNorm = ", sn_ft.shape, sn_ft.dtype)
        print("skeleton = ", sk_ft.shape, sk_ft.dtype)

    hospisign_labels = get_hospisign_labels(nos=nos, sortBy=labelsSortBy)
    labels = hospisign_labels["labels"]
    labels_sui = hospisign_labels["labels_sui"]
    label_map = hospisign_labels["label_map"]

    if verbose > 0:
        print("labels_sui = ", labels_sui.shape, labels_sui.dtype)
        print("labels = ", labels.shape, type(labels))
    ft = {
        "hg": hg_ft,
        "sn": sn_ft,
        "sk": sk_ft,
    }
    lab = {
        "labels_sui": labels_sui,
        "labels": labels,
        "label_map": label_map,
    }
    return ft, lab

def combine_pca_hospisign_data(dataIdent, pca_dim=256, nos=11, verbose=2):
    ft, lab = get_hospisign_feats(nos=nos, verbose=verbose)

    print('hg_ft.shape = ', ft["hg"].shape, ', sn_ft.shape = ', ft["sn"].shape, ', sk_ft.shape = ', ft["sk"].shape)

    if verbose > 0:
        print("hg - min(", ft["hg"].min(), "), max(", ft["hg"].max(), ")")
        print("sn - min(", ft["sn"].min(), "), max(", ft["sn"].max(), ")")
        print("sk - min(", ft["sk"].min(), "), max(", ft["sk"].max(), ")")

    nan_sk, nan_sn, nan_hg = list(), list(), list()
    for i in range(ft["sk"].shape[0]):
        if (np.isnan(ft["sk"][i]).any()):
            nan_sk.append(i)
            ft["sk"][i] = np.zeros(ft["sk"][i].shape)  # print(i,":",ft["sk"][i]) #print(ft["sk"][i].shape)
        if (np.isnan(ft["sn"][i]).any()):
            nan_sn.append(i)
            ft["sn"][i] = np.zeros(ft["sn"][i].shape)  # print(i,":",ft["sn"][i]) #print(ft["sn"][i].shape)
        if (np.isnan(ft["hg"][i]).any()):
            nan_hg.append(i)
            ft["hg"][i] = np.zeros(ft["hg"][i].shape)  # print(i,":",ft["hg"][i]) #print(ft["hg"][i].shape)

    if nan_hg:
        nan_hg = np.squeeze(np.vstack(nan_hg))
        if verbose > 0:
            print("nan_hg: ", nan_hg)
    if nan_sn:
        nan_sn = np.squeeze(np.vstack(nan_sn))
        if verbose > 0:
            print("nan_sn: ", nan_sn)
    if nan_sk:
        nan_sk = np.squeeze(np.vstack(nan_sk))
        if verbose > 0:
            print("nan_sk: ", nan_sk)

    if (dataIdent == "hog" or dataIdent == "hg"):
        feats = ft["hg"]
    elif (dataIdent == "skeleton" or dataIdent == "sk"):
        feats = ft["sk"]
    elif (dataIdent == "snv" or dataIdent == "sn"):
        feats = ft["sn"]
    elif (dataIdent == "hgsk"):
        feats = np.concatenate([ft["hg"].T, ft["sk"].T]).T
    elif (dataIdent == "hgsn"):
        feats = np.concatenate([ft["hg"].T, ft["sn"].T]).T
    elif (dataIdent == "snsk"):
        feats = np.concatenate([ft["sn"].T, ft["sk"].T]).T
    elif (dataIdent == "hgsnsk"):
        feats = np.concatenate([ft["hg"].T, ft["sn"].T, ft["sk"].T]).T

    feats_pca, exp_var_rat = funcH.applyMatTransform(feats, applyPca=True, whiten=True, normMode="", verbose=verbose)
    feats = feats_pca[:, 0:pca_dim]
    print(dataIdent, '.shape = ', feats.shape, ' loaded.')

    return feats, lab["labels"], lab["labels_sui"], lab["label_map"]

def get_result_table_out(result_file_name_full, class_names):
    a = np.load(result_file_name_full)
    print(a.files)

    dataIdent_ = a["dataIdent_"]
    testUser_ = a["testUser_"]
    validUser_ = a["validUser_"]
    hid_state_cnt_vec_ = a["hid_state_cnt_vec_"]

    accvectr = a["accvectr_"]
    accvecva = a["accvecva_"]
    accvecte = a["accvecte_"]
    bestVaID = np.argmax(accvecva)
    bestTeID = np.argmax(accvecte)
    formatStr = "5.3f"
    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID,accvecva[bestTeID],accvecte[bestTeID]))
    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1], accvecte[-1]))
    print("accvectr_=", accvectr.shape, ", accvecva_=", accvecva.shape, ", accvecte_=", accvecte.shape)

    preds_best_ = np.squeeze(a["preds_best_"])
    labels_best_ = np.squeeze(a["labels_best_"])
    print("preds_best_=", preds_best_.shape, ", labels_best_=", labels_best_.shape)

    # check if exist. if yes go on
    # uniqLabs = np.unique(labels_best_)
    # classCount = len(uniqLabs)
    # print("uniqLabs=", uniqLabs, ", classCount_=", classCount)

    conf_mat_ = confusion_matrix(labels_best_, preds_best_)
    print(conf_mat_.shape, class_names.shape)
    saveConfFileName_full = result_file_name_full.replace('.npz', '.png')
    if not os.path.exists(saveConfFileName_full):
        print("saving--", saveConfFileName_full)
        try:
            funcH.plot_confusion_matrix(conf_mat_, class_names=class_names, confusionTreshold=0.2, show_only_confused=True, saveConfFigFileName=saveConfFileName_full)
        except:
            pass

    confMatStats, df_slctd_table = funcH.calcConfusionStatistics(conf_mat_, categoryNames=class_names,
                                                                 selectedCategories=None, verbose=0)
    df_slctd_table = df_slctd_table.sort_values(["F1_Score"], ascending=False)
    print(df_slctd_table)
    saveF1TableFileName_full = result_file_name_full.replace('.npz', '_F1.csv')
    if not os.path.exists(saveF1TableFileName_full):
        print("saving--", saveF1TableFileName_full)
        df_slctd_table.to_csv(saveF1TableFileName_full)
    else:
        print("already saved--", saveF1TableFileName_full)

    print("dataIdent_=", dataIdent_, "\ntestUser_=", testUser_, "\nvalidUser_=", validUser_, "\nhid_state_cnt_vec_=", hid_state_cnt_vec_)
    print(("bestVaID({:" + formatStr + "}),trAcc({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID,accvectr[bestVaID],accvecva[bestVaID],accvecte[bestVaID]))
    return df_slctd_table

# study_sihouette_better
def update_centroid_df(centroid_df, cluster_ids):
    uq_pr = np.unique(cluster_ids)
    num_of_samples = []
    for i in range(len(uq_pr)):
        cluster_id = uq_pr[i]
        cluster_inds = funcH.getInds(cluster_ids, i)
        num_of_samples.append(len(cluster_inds))
    centroid_df['num_of_samples'] = num_of_samples
    return centroid_df

def analyze_correspondance_results(correspondance_tuple, centroid_df, pred_vec, lab_vec):
    df = pd_df({'labels': lab_vec[np.asarray(centroid_df['sampleID'], dtype=int)],
                'klusterID': np.asarray(centroid_df['klusterID'], dtype=int),
                'sampleCounts': np.asarray(centroid_df['num_of_samples'], dtype=int)})
    print('correspondance results:')
    print(df.groupby(['labels'])[['labels', 'sampleCounts']].sum())
    corr_in_clust = pred_vec[correspondance_tuple[0]]
    corr_ou_clust = pred_vec[correspondance_tuple[1]]
    _confMat_corr_preds = confusion_matrix(corr_in_clust, corr_ou_clust)
    acc_corr_preds = 100 * np.sum(np.diag(_confMat_corr_preds)) / np.sum(
        np.sum(_confMat_corr_preds))
    print("_confMat_corr_preds - acc({:6.4f})".format(acc_corr_preds))

    corr_in_labels = lab_vec[correspondance_tuple[0]]
    corr_ou_labels = lab_vec[correspondance_tuple[1]]
    _confMat_corr = confusion_matrix(corr_in_labels, corr_ou_labels)
    acc_corr = 100 * np.sum(np.diag(_confMat_corr)) / np.sum(np.sum(_confMat_corr))
    print("confMat - acc({:6.4f}), correspondance match:\n".format(acc_corr), pd_df(_confMat_corr))

def analyze_reconstruction_values(reconstruction_loss_vec, cluster_labels, real_labels,
                                  centroid_info_pdf=None, label_names=None, verbose=0,
                                  figsize=(12, 7), dpi=360, lw=[5, 4],
                                  show_title=True, str_deg=45, str_size=12,
                                  save_at_file_name=''):
    # map loss to 0 1 area
    rec_los_sorted, rec_idx = funcH.sortVec(-reconstruction_loss_vec)
    rec_los_sorted_0_1 = funcH.map_0_1(-rec_los_sorted)
    # map clusters to class labels by centroids
    predictions_mapped = funcH.map_predictions(real_labels, cluster_labels, centroid_info_pdf=centroid_info_pdf)

    cumsum_preds_rec = funcH.cumsum_preds(real_labels, predictions_mapped, rec_idx)
    data_perc_vec = np.arange(0, len(cumsum_preds_rec)) / len(cumsum_preds_rec)

    plt.close('all')
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.plot(data_perc_vec, cumsum_preds_rec, lw=lw[0], label='rec_cs', color='red', ls='-', zorder=0)
    ax.plot(data_perc_vec, rec_los_sorted_0_1, lw == lw[1], label='rec_los_sorted_0_1', color='orange', ls='-',
            zorder=0)
    plt.legend(loc='upper right', prop={'size': str_size})

    # data_id_at_i = np.argmax(rec_los_sorted_0_1 > 0.1) - 1
    # data_p = data_id_at_i / len(cumsum_preds_rec)
    # ac_at = cumsum_preds_rec[data_id_at_i]
    data_p = 0.2
    data_id_at_i = int(len(cumsum_preds_rec) * data_p)
    # rec_v = rec_los_sorted_0_1[data_id_at_i]
    ac_at = cumsum_preds_rec[data_id_at_i]
    acc_str = 'at({:2.1f})_acc({:4.2f})_min<{:4.2f}>_mean<{:4.2f}>'.format(data_p, 100 * ac_at,
                                                                           100 * np.min(cumsum_preds_rec),
                                                                           100 * np.mean(cumsum_preds_rec))

    title_str = acc_str if show_title else ''
    ax.set(xlabel='data percentage', ylabel='accuracy', title=title_str)

    ax.set_xlim([0, 1.0])
    ax.set_xticks(np.arange(0, 1.00, 0.1), minor=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    ax.set_ylim([0.0, 1.0])
    ax.set_yticks(np.arange(-0.2, 1.00, 0.1), minor=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    ax.title.set_position([.5, 1.05])
    ax.grid()
    plt.legend(loc='center left')

    print("*-*-*")
    data_p_last = 0.0
    for i in range(9):
        rec_v = i * 0.1  # rec_los_sorted_0_1 value
        data_id_at_i = np.argmax(rec_los_sorted_0_1 > rec_v) - 1
        data_p = data_id_at_i / len(cumsum_preds_rec)
        if data_p - data_p_last < 0.01:
            continue
        data_p_last = data_p
        ac_at = cumsum_preds_rec[data_id_at_i]
        acc_ident = "4.2f"
        if ac_at == 1.00:
            acc_ident = "3.0f"
        if verbose > 0:
            print(str("rec_v({:.1f}),at({:d}),acc({:" + acc_ident + "})").format(rec_v, data_id_at_i, 100 * ac_at))
        _x = data_id_at_i / len(cumsum_preds_rec)
        ax.text(x=_x, y=rec_los_sorted_0_1[data_id_at_i],
                s=str("s({:2.1f})dp({:2.1f})acc({:" + acc_ident + "})*").format(rec_v, data_p, 100 * ac_at),
                va='top', ha='right',
                color="green", fontsize=str_size, rotation=str_deg)
        ax.plot(np.asarray([_x, _x]), np.asarray([rec_los_sorted_0_1[data_id_at_i], ac_at]), lw=3,
                color='green', ls='-', zorder=0)
    print("*-*-*")
    for i in range(1, 9):
        _dp = i * 0.1
        data_id_at_i = int(len(cumsum_preds_rec) * _dp)
        rec_v = rec_los_sorted_0_1[data_id_at_i]
        ac_at = cumsum_preds_rec[data_id_at_i]
        if verbose > 0:
            print("data_perc({:.1f}),at({:d}),acc({:6.4f})".format(_dp, data_id_at_i, 100 * ac_at))
        ax.text(x=data_id_at_i / len(cumsum_preds_rec), y=ac_at,
                s="dp({:2.1f})rv({:3.2f})acc({:6.4f})".format(_dp, rec_v, 100 * ac_at),
                va='bottom', ha='left',
                color="blue", fontsize=str_size, rotation=str_deg)

    if len(save_at_file_name) > 0:
        fig.savefig(save_at_file_name.replace('ACCSTR', acc_str))

    result_dict = {
        "preds_sorted": cumsum_preds_rec,
        "data_perc_vec": data_perc_vec,
    }
    return result_dict

def analyze_silhouette_values(sample_silhouette_values, cluster_labels, real_labels,
                              centroid_info_pdf=None, label_names=None, conf_plot_save_to='', verbose=0,
                              figsize=(24, 12), lw=[5, 4, 3], show_title=True, str_deg=45, str_size='x-large'):
    _confMat, kluster2Classes, kr_pdf, _, _ = funcH.countPredictionsForConfusionMat(real_labels, cluster_labels, centroid_info_pdf=centroid_info_pdf)
    sampleCount = np.sum(np.sum(_confMat))
    acc_doga_base = 100 * np.sum(np.diag(_confMat)) / sampleCount
    if verbose > 0:
        print("accuracy for {:d} clusters = {:4.3f}".format(len(np.unique(cluster_labels)), acc_doga_base))

    mapped_class_vec = np.array(kluster2Classes)[:, 1].squeeze()
    predictions_mapped, mappedKlustersSampleCnt = funcH.getMappedKlusters(cluster_labels, mapped_class_vec)
    np.sum(predictions_mapped == real_labels) / len(predictions_mapped)

    plot_title_str = 'before silhouette - '
    funcH.plot_confusion_matrix(_confMat.T, class_names=label_names, saveConfFigFileName=conf_plot_save_to, plot_title_str=plot_title_str, verbose=verbose)
    try:
        funcH.plot_confusion_matrix(_confMat.T, class_names=label_names, saveConfFigFileName=conf_plot_save_to.replace(".png","_confused.png"), plot_title_str=plot_title_str, show_only_confused=True, verbose=verbose)
    except:
        pass

    sample_silhouette_values_sorted, idx = funcH.sortVec(sample_silhouette_values)
    labels_sorted = real_labels[idx]
    cluster_labels_sorted = cluster_labels[idx]
    preds_sorted = predictions_mapped[idx]
    all_ones = np.ones(preds_sorted.shape)
    pred_cumsum = np.cumsum(preds_sorted == labels_sorted) / np.cumsum(all_ones)
    save_acc_silhouette_fig_file_name = conf_plot_save_to.replace("_conf_", "_acc_silhouette_")
    data_perc_vec = np.arange(0, len(pred_cumsum)) / len(pred_cumsum)

    first_neg_sample_id = np.argmax(sample_silhouette_values_sorted < 0.00)-1
    accuracy_at_last_pos = pred_cumsum[first_neg_sample_id]
    if verbose > 1:
        print("first_neg_sample_id(", str(first_neg_sample_id), ") accuracy_at_last_pos(", "{:4.2f}".format(100*accuracy_at_last_pos) ,")", end=',')
    centroid_info_pdf_new = centroid_info_pdf.copy()
    clusters_to_remove = []
    samples_to_remove = []
    for r in range(len(centroid_info_pdf_new)):
        old_index = centroid_info_pdf_new["sampleID"].values[r]
        if verbose > 2:
            print("\n old centroid index(", old_index, ") changed to new index(", end='')
        centroid_info_pdf_new["sampleID"].values[r] = np.argmax(idx == old_index)
        if verbose > 2:
            print(centroid_info_pdf_new["sampleID"].values[r], "),", end='')
        if centroid_info_pdf_new["sampleID"].values[r] > first_neg_sample_id:
            clusters_to_remove.append(r)
            if verbose > 2:
                print("this row will be dropped("+str(r)+")", end='')
            klusterID_to_remove = centroid_info_pdf_new["klusterID"].values[r]
            samples_to_remove_cur = funcH.getInds(cluster_labels_sorted,klusterID_to_remove)
            samples_to_remove.append(np.asarray(samples_to_remove_cur).squeeze())

    valid_sample_cnt = 0
    if len(samples_to_remove) > 0:
        samples_to_remove = np.concatenate(np.asarray(samples_to_remove)).squeeze()
        valid_sample_cnt = np.sum(samples_to_remove<first_neg_sample_id)
        labels_sorted = np.delete(labels_sorted, samples_to_remove)
        cluster_labels_sorted = np.delete(cluster_labels_sorted, samples_to_remove)
        centroid_info_pdf_new = centroid_info_pdf_new.drop(np.asarray(clusters_to_remove))
        for r in range(len(centroid_info_pdf_new)):
            new_index = centroid_info_pdf_new["sampleID"].values[r]
            before_sample_cnt = np.sum(samples_to_remove < new_index)
            newer_index = new_index-before_sample_cnt
            if new_index != newer_index:
                if verbose > 2:
                    print("\nnew centroid index(", new_index, ") changed to newer index(", end='')
                centroid_info_pdf_new["sampleID"].values[r] = newer_index
                if verbose > 2:
                    print(centroid_info_pdf_new["sampleID"].values[r], ")", end='')

    confMat_new, _, _, _, _ = funcH.countPredictionsForConfusionMat(labels_sorted[:first_neg_sample_id-valid_sample_cnt], cluster_labels_sorted[:first_neg_sample_id-valid_sample_cnt],
                                                                              centroid_info_pdf=centroid_info_pdf_new)
    title_str = "first_neg_at " + str(first_neg_sample_id) + "(" + "{:4.2f}".format(100*first_neg_sample_id/sampleCount) + ")\n"
    title_str += str(sampleCount - first_neg_sample_id) + " samples to remove\n"
    title_str += 'old_accuracy<{:4.2f}>_new'.format(acc_doga_base)
    funcH.plot_confusion_matrix(confMat_new.T, class_names=label_names, saveConfFigFileName=conf_plot_save_to.replace("_conf_", "_conf_post_silhouette_"), plot_title_str=title_str, verbose=verbose)
    try:
        funcH.plot_confusion_matrix(confMat_new.T, class_names=label_names, saveConfFigFileName=conf_plot_save_to.replace("_conf_", "_conf_post_silhouette_").replace(".png","_confused.png"), plot_title_str=title_str, show_only_confused=True, verbose=verbose)
    except:
        pass

    plt.close('all')
    fig, ax = plt.subplots(1, figsize=figsize, dpi=180)
    ax.plot(data_perc_vec, pred_cumsum, lw=lw[0], label='accuracy', color='blue', ls='-', zorder=0)
    ax.plot(data_perc_vec, sample_silhouette_values_sorted, lw=lw[1], label='silhouette_prec', color='green', ls='-', zorder=0)
    ax.plot(np.asarray([0, 1]), np.asarray([accuracy_at_last_pos, accuracy_at_last_pos]), lw=lw[2], label='first_neg_sample', color='red', ls='-', zorder=0)
    # Data for plotting

    title_str += '_accuracy<{:4.2f}>'.format(100*accuracy_at_last_pos)
    title_str = title_str if show_title else ''
    ax.set(xlabel='data percentage', ylabel='accuracy', title=title_str)

    ax.set_xlim([0, 1.0])
    ax.set_xticks(np.arange(0, 1.00, 0.1), minor=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    ax.set_ylim([-0.2, 1.0])
    ax.set_yticks(np.arange(-0.2, 1.00, 0.1), minor=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    ax.title.set_position([.5, 1.05])
    ax.grid()
    plt.legend(loc='lower left')

    print("*-*-*")
    data_p_last = 0.0
    for i in range(9):
        sil_v = 0.90 - i * 0.1  # silhouette value
        data_id_at_i = np.argmax(sample_silhouette_values_sorted < sil_v) - 1
        data_p = data_id_at_i/len(pred_cumsum)
        if data_p-data_p_last < 0.01:
            continue
        data_p_last = data_p
        ac_at = pred_cumsum[data_id_at_i]
        acc_ident = "4.2f"
        if ac_at == 1.00:
            acc_ident = "3.0f"
        if verbose > 0:
            print(str("sil_v({:.1f}),at({:d}),acc({:"+acc_ident+"})").format(sil_v, data_id_at_i, 100*ac_at))
        _x = data_id_at_i/len(pred_cumsum)
        ax.text(x=_x, y=sample_silhouette_values_sorted[data_id_at_i],
                s=str("s({:2.1f})dp({:2.1f})acc({:"+acc_ident+"})*").format(sil_v, data_p, 100*ac_at),
                va='top', ha='right',
                color="green", fontsize=str_size, rotation=str_deg)
        ax.plot(np.asarray([_x, _x]), np.asarray([sample_silhouette_values_sorted[data_id_at_i], ac_at]), lw=3,
                color='green', ls='-', zorder=0)
    print("*-*-*")
    for i in range(1, 9):
        _dp = i * 0.1
        data_id_at_i = int(len(pred_cumsum) * _dp)
        sil_v = sample_silhouette_values_sorted[data_id_at_i]
        ac_at = pred_cumsum[data_id_at_i]
        if verbose > 0:
            print("data_perc({:.1f}),at({:d}),acc({:6.4f})".format(_dp, data_id_at_i, 100*ac_at))
        ax.text(x=data_id_at_i/len(pred_cumsum), y=ac_at,
                s="dp({:2.1f})sv({:2.1f})acc({:6.4f})".format(_dp, sil_v, 100*ac_at),
                va='bottom', ha='left',
                color="blue", fontsize=str_size, rotation=str_deg)

    fig.savefig(save_acc_silhouette_fig_file_name)

    result_dict = {
        "_confMat": _confMat,
        "confMat_new": confMat_new,
        "mapped_class_vec": mapped_class_vec,
        "preds_sorted": preds_sorted,
        "labels_sorted": labels_sorted,
        "data_perc_vec": data_perc_vec,
    }
    return result_dict

def calc_tuple_score_vals(tuple_score_sum, lab_vec, cor_tup, sort_ascend=False, _def_str="ds"):
    sort_mul = 2 * (float(sort_ascend) - 0.5)
    tuple_idx = np.argsort(sort_mul * tuple_score_sum)
    tup_sor_a_idx = cor_tup[0][tuple_idx]
    tup_sor_b_idx = cor_tup[1][tuple_idx]
    lab_vec_a = lab_vec[tup_sor_a_idx]
    lab_vec_b = lab_vec[tup_sor_b_idx]
    _cn_a = []
    _cn_b = []
    uniq_class_cnt_perc_a = np.zeros(lab_vec_a.shape, dtype=float)
    uniq_class_cnt_perc_b = np.zeros(lab_vec_b.shape, dtype=float)
    n = len(lab_vec)
    for i in range(n):
        if lab_vec_a[i] not in _cn_a:
            _cn_a.append(lab_vec_a[i])
        if lab_vec_b[i] not in _cn_b:
            _cn_b.append(lab_vec_b[i])
        uniq_class_cnt_perc_a[i] = len(_cn_a)
        uniq_class_cnt_perc_b[i] = len(_cn_b)
    print("general_acc_for(" + _def_str + "):", np.sum(lab_vec_a == lab_vec_b) / len(lab_vec_a))
    pred_cumsum = np.cumsum(lab_vec_a == lab_vec_b) / np.cumsum(lab_vec_b == lab_vec_b)
    max_run_acc_idx = np.argmax(pred_cumsum)
    max_run_acc = pred_cumsum[max_run_acc_idx]
    print("max_run_acc({:6.4f}), at {:d}(%{:4.2f})".format(max_run_acc, max_run_acc_idx, max_run_acc_idx / n))

    cnt_lab_uniq = len(np.unique(lab_vec))
    cpa = uniq_class_cnt_perc_a / cnt_lab_uniq
    cpb = uniq_class_cnt_perc_b / cnt_lab_uniq
    print(cpa)

    return pred_cumsum, tuple_score_sum[tuple_idx], cpa, cpb
def calc_tup_sc_plot_01(sil_sort_pred_cumsum, sil_tup_sum_sorted, cpa_sil, cpb_sil, _s='sil_', figsize=(12, 8), dpi=600):
    data_perc_vec = np.arange(0, len(sil_sort_pred_cumsum)) / len(sil_sort_pred_cumsum)
    sil_tup_sum_sorted_n = funcH.map_0_1(sil_tup_sum_sorted)
    plt.close('all')
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    ax.plot(data_perc_vec, sil_sort_pred_cumsum, lw=2, label=_s + 'sort', color='blue', ls='-', zorder=0)
    ax.plot(data_perc_vec, sil_tup_sum_sorted_n, lw=2, label=_s + 'tup_sum_sorted', color='cyan', ls='-', zorder=0)
    ax.plot(data_perc_vec, cpa_sil, lw=3, label=_s + 'cpa', color='purple', ls='-', zorder=0)
    ax.plot(data_perc_vec, cpb_sil, lw=1, label=_s + 'cpb', color='orange', ls='-', zorder=0)
    plt.legend(loc='lower left')
    plt.show()
def calc_tup_sc(sil_vals, reconstruction_loss, cor_tup, lab_vec, ep_id=None, figsize=(12, 8), dpi=600, experiments_folder=''):
    n = len(sil_vals)
    tuple_sihouette_score_sum = np.asarray([sil_vals[cor_tup[0][i]] + sil_vals[cor_tup[1][i]] for i in range(n)])
    rec_los_0_1 = 1 - funcH.map_0_1(reconstruction_loss)
    tuple_rec_score_sum = np.asarray([rec_los_0_1[cor_tup[0][i]] + rec_los_0_1[cor_tup[1][i]] for i in range(n)])
    tuple_sr_score_sum = tuple_sihouette_score_sum + tuple_rec_score_sum

    sil_sort_pred_cumsum, sil_tup_sum_sorted, cpa_sil, cpb_sil = calc_tuple_score_vals(tuple_sihouette_score_sum,
                                                                                       lab_vec, cor_tup,
                                                                                       sort_ascend=False,
                                                                                       _def_str="sil_sort")
    rec_sort_pred_cumsum, rec_tup_sum_sorted, cpa_rec, cpb_rec = calc_tuple_score_vals(tuple_rec_score_sum, lab_vec,
                                                                                       cor_tup, sort_ascend=False,
                                                                                       _def_str="rec_sort")
    sr_sort_pred_cumsum, sr_tup_sum_sorted, cpa_sr, cpb_sr = calc_tuple_score_vals(tuple_sr_score_sum, lab_vec, cor_tup,
                                                                                   sort_ascend=False,
                                                                                   _def_str="sr_sort")

    calc_tup_sc_plot_01(sil_sort_pred_cumsum, sil_tup_sum_sorted, cpa_sil, cpb_sil, _s='sil_')
    calc_tup_sc_plot_01(rec_sort_pred_cumsum, rec_tup_sum_sorted, cpa_rec, cpb_rec, _s='rec_')
    calc_tup_sc_plot_01(sr_sort_pred_cumsum, sr_tup_sum_sorted, cpa_sr, cpb_sr, _s='sr_')

    plt.close('all')
    fig, ax = plt.subplots(1, figsize=figsize, dpi=dpi)
    data_perc_vec = np.arange(0, len(sil_sort_pred_cumsum)) / len(sil_sort_pred_cumsum)
    ax.plot(data_perc_vec, sil_sort_pred_cumsum, lw=1, label='silhouette_sort', color='red', ls='-', zorder=0)
    ax.plot(data_perc_vec, rec_sort_pred_cumsum, lw=1, label='reconstruction_sort', color='blue', ls='-', zorder=0)
    ax.plot(data_perc_vec, sr_sort_pred_cumsum, lw=2, label='sil+rec_sort', color='purple', ls='-', zorder=0)
    ax.set_ylim([0.75, 1.0])
    ax.set_xlim([0.0, 1.0])
    title_str = "minacc({:6.4f}),meanacc({:6.4f})".format(np.min(rec_sort_pred_cumsum), np.mean(rec_sort_pred_cumsum))
    plt.title(title_str)
    plt.legend(loc='lower left')
    if ep_id is not None and experiments_folder!='':
        min_acc = "{:6.4f}".format(np.min(rec_sort_pred_cumsum))
        exp_fold = experiments_folder
        saveFileName = os.path.join(exp_fold, "plots", "compare{:03d}_{}.jpeg".format(ep_id, min_acc))
        print("*-*-*Saving({:})".format(saveFileName))
        plt.savefig(saveFileName)
    else:
        print("*-*-*ep_id({:}, experiments_folder({:})".format(ep_id, experiments_folder))
    plt.show()