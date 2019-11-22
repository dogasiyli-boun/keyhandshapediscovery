import helperFuncs as funcH
import dataLoaderFuncs as funcD
import numpy as np
import os
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import time
import datetime

def createExperimentName(trainParams, modelParams, rnnParams):

    pcaCountStr = str(modelParams["pcaCount"]) if modelParams["pcaCount"] > 0 else "Feats"

    if modelParams["trainMode"] == "corsa":
        exp_name  = str(modelParams["trainMode"]) + \
                    '_pd' + str(modelParams["posterior_dim"]) + \
                    '_wr' + str(modelParams["weight_of_regularizer"]) + \
                    '_' + str(modelParams["dataToUse"]) + pcaCountStr + '_' + str(modelParams["numOfSigns"]) + \
                    '_bs' + str(trainParams["batch_size"]) + \
                    '_dM' + str(rnnParams["dataMode"]) + \
                    '_ts' + str(rnnParams["timesteps"]) + \
                    '_cp' + str(trainParams["applyCorr"]) + \
                    '_cRM' + str(trainParams["corr_randMode"])
        if rnnParams["dropout"] > 0:
            exp_name += '_do' + str(rnnParams["dropout"])
    elif modelParams["trainMode"] == "rsa":
        exp_name  = str(modelParams["trainMode"]) + \
                    '_pd' + str(modelParams["posterior_dim"]) + \
                    '_wr' + str(modelParams["weight_of_regularizer"]) + \
                    '_' + str(modelParams["dataToUse"]) + pcaCountStr + '_' + str(modelParams["numOfSigns"]) + \
                    '_bs' + str(trainParams["batch_size"]) + \
                    '_dM' + str(rnnParams["dataMode"]) + \
                    '_ts' + str(rnnParams["timesteps"])
        if rnnParams["dropout"] > 0:
            exp_name += '_do' + str(rnnParams["dropout"])
        if rnnParams["dataMode"] == 1:
            exp_name += '_pc' + str(rnnParams["patchFromEachVideo"])
        if rnnParams["dataMode"] == 2:
            exp_name += '_fo' + str(rnnParams["frameOverlap"])
    elif modelParams["trainMode"] == "cosae":
        exp_name  = str(modelParams["trainMode"]) + \
                    '_pd' + str(modelParams["posterior_dim"]) + \
                    '_wr' + str(modelParams["weight_of_regularizer"]) + \
                    '_' + str(modelParams["dataToUse"]) + pcaCountStr + '_' + str(modelParams["numOfSigns"]) + \
                    '_bs' + str(trainParams["batch_size"]) + \
                    '_cp' + str(trainParams["applyCorr"]) + \
                    '_cRM' + str(trainParams["corr_randMode"])
    elif modelParams["trainMode"] == "sae":
        exp_name  = str(modelParams["trainMode"]) + \
                    '_pd' + str(modelParams["posterior_dim"]) + \
                    '_wr' + str(modelParams["weight_of_regularizer"]) + \
                    '_' + str(modelParams["dataToUse"]) + pcaCountStr + '_' + str(modelParams["numOfSigns"]) + \
                    '_bs' + str(trainParams["batch_size"])
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

def createPCAOfData(data_dir, dataToUse, sign_count, recreate=False):
    npy_PCAFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=sign_count, expectedFileType='PCA', pcaCount=-1)
    npy_PCAFileName = os.path.join(data_dir, npy_PCAFileName)
    if os.path.isfile(npy_PCAFileName) and not recreate:
        feats_pca = np.load(npy_PCAFileName)
        exp_var_rat = []
        print('loaded ', dataToUse, 'Feats(', feats_pca.shape, ') from : ', npy_PCAFileName)
        print('Max of featsPCA = ', np.amax(feats_pca), ', Min of featsPCA = ', np.amin(feats_pca))
    else:
        npy_FeatsFileName = funcD.getFileName(dataToUse, sign_count, expectedFileType='Data', pcaCount=-1)
        npy_FeatsFileName = os.path.join(data_dir, npy_FeatsFileName)
        feats = np.load(npy_FeatsFileName)
        print('Max of feats = ', np.amax(feats), ', Min of feats = ', np.amin(feats))
        feats_pca, exp_var_rat = funcH.applyMatTransform(feats, applyNormalization=True, applyPca=True, whiten=True, verbose=2)
        np.save(npy_PCAFileName, feats_pca)
    return feats_pca, exp_var_rat

def convert_Mat2NPY(dataToUse, data_dir, signCnt, featureStr, labelStr, possible_fname_init, recreate=False):
    npy_labels_file_name = funcD.getFileName(dataToUse=dataToUse, numOfSigns=signCnt, expectedFileType='Labels')
    npy_labels_file_name = os.path.join(data_dir, npy_labels_file_name)
    npy_feats_file_name = funcD.getFileName(dataToUse=dataToUse, numOfSigns=signCnt, expectedFileType='Data')
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

def createPCADimsOfData(data_dir, data2use, sign_count, dimArray = [256, 512, 1024], recreate=False):
    featsPCA, exp_var_rat = createPCAOfData(data_dir=data_dir, dataToUse=data2use, sign_count=sign_count, recreate=recreate)
    try:
        print(data2use, ' feats exp_var_rat[', list(map('{:.2f}%'.format, dimArray)), '] = ', list(map('{:.2f}%'.format, exp_var_rat[dimArray])))
    except:
        pass

    for dims in dimArray:
        npy_PCAFileName = funcD.getFileName(data2use, sign_count, expectedFileType='Data', pcaCount=dims)
        npy_PCAFileName = os.path.join(data_dir, npy_PCAFileName)
        featsToSave = featsPCA[:,0:dims]
        if os.path.isfile(npy_PCAFileName) and not recreate:
            featsToSave = np.load(npy_PCAFileName)
            print('pca exists at : ', npy_PCAFileName)
        else:
            print("features.shape:", featsToSave.shape)
            print('saving pca sn features at : ', npy_PCAFileName)
            np.save(npy_PCAFileName, featsToSave)

def runClusteringOnFeatSet(data_dir, results_dir, dataToUse, numOfSigns, pcaCount, expectedFileType, clusterModels = ['Kmeans', 'GMM_diag', 'GMM_full', 'Spectral'], randomSeed=5):
    seed(randomSeed)
    tf.set_random_seed(seed=randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)

    featsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType=expectedFileType, pcaCount=pcaCount) # 'hogFeats_41.npy', 'skeletonFeats_41.npy'
    detailedLabelsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='DetailedLabels', pcaCount=pcaCount) # 'detailedLabels_41.npy'
    detailedLabelsFileNameFull = data_dir + os.sep + detailedLabelsFileName
    labelsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='Labels', pcaCount=pcaCount) # 'labels_41.npy'
    labelsFileNameFull = data_dir + os.sep + labelsFileName

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='BaseResultName', pcaCount=pcaCount)
    funcH.createDirIfNotExist(os.path.join(results_dir, 'baseResults'))
    baseResultFileNameFull = os.path.join(results_dir, 'baseResults', baseResultFileName)

    featSet = np.load(data_dir + os.sep + featsFileName)
    detailedLabels_all = np.load(detailedLabelsFileNameFull)
    labels_all = np.load(labelsFileNameFull)
    non_zero_labels = labels_all[np.where(labels_all)]

    print('*-*-*-*-*-*-*running for : ', featsFileName, '*-*-*-*-*-*-*')
    print('featSet(', featSet.shape, '), detailedLabels(', detailedLabels_all.shape, '), labels_All(', labels_all.shape, '), labels_nonzero(', non_zero_labels.shape, ')')

    clustCntVec = [32, 64, 128, 256, 512]
    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = list(np.load(baseResultFileNameFull, allow_pickle=True))
    else:
        resultDict = []

    headerStrFormat = "+++frmfile(%15s) clusterModel(%8s), clusCnt(%4s)"
    valuesStrFormat = "nmiAll(%.2f) * accAll(%.2f) * nmiNoz(%.2f) * accNoz(%.2f) * emptyClusters(%d)"

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
            predictionFileName = baseResultFileName.replace("_baseResults.npy","") + "_" + clusterModel + "_" + str(curClustCnt) + ".npz"
            predictionFileNameFull = os.path.join(results_dir, 'baseResults', predictionFileName)
            predictionFileExist = os.path.isfile(predictionFileNameFull)
            if not foundResult or not predictionFileExist:
                if foundResult and not predictionFileExist:
                    print('running again for saving predictions')
                t = time.time()
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(featsFileName, 'clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ') running.')
                predClusters = funcH.clusterData(featVec=featSet, n_clusters=curClustCnt, applyNormalization=False, applyPca=False, clusterModel=clusterModel)
                print('elapsedTime(', time.time() - t, ')')

                nmi_cur, acc_cur = funcH.get_NMI_Acc(labels_all, predClusters)

                non_zero_preds = predClusters[np.where(labels_all)]
                nmi_cur_nz, acc_cur_nz = funcH.get_NMI_Acc(non_zero_labels, non_zero_preds)

                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)

                resultList = [featsFileName.replace('.npy', ''), clusterModel, curClustCnt, [nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)

                print(valuesStrFormat % (nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins))
                print(resultList[3][5][0:10])
                np.save(baseResultFileNameFull, resultDict, allow_pickle=True)
                np.savez(predictionFileNameFull, labels_all, predClusters)

    np.set_printoptions(prevPrintOpts)
    return resultDict

def runOPTICSClusteringOnFeatSet(data_dir, results_dir, dataToUse, numOfSigns, pcaCount, expectedFileType, clustCntVec = [32, 64, 128, 256, 512], randomSeed=5, updateResultBaseFile=False):
    seed(randomSeed)
    tf.set_random_seed(seed=randomSeed)
    prevPrintOpts = np.get_printoptions()
    np.set_printoptions(precision=4, suppress=True)

    featsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType=expectedFileType, pcaCount=pcaCount) # 'hogFeats_41.npy', 'skeletonFeats_41.npy'
    detailedLabelsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='DetailedLabels', pcaCount=pcaCount) # 'detailedLabels_41.npy'
    detailedLabelsFileNameFull = data_dir + os.sep + detailedLabelsFileName
    labelsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='Labels', pcaCount=pcaCount) # 'labels_41.npy'
    labelsFileNameFull = data_dir + os.sep + labelsFileName

    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='BaseResultName', pcaCount=pcaCount)
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
                predClusters = funcH.clusterData(featVec=featSet, n_clusters=curClustCnt, applyNormalization=False, applyPca=False, clusterModel=clusterModel)
                print('elapsedTime(', time.time() - t, ')')
                np.savez(predictionFileNameFull, labels_all, predClusters)
            else:
                npz = np.load(predictionFileNameFull)
                if 'arr_0' in npz.files:
                    predClusters = npz['arr_1']
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

def loadData(model_params, numOfSigns, data_dir):
    featsFileName = funcD.getFileName(dataToUse=model_params["dataToUse"], numOfSigns=numOfSigns,
                                      expectedFileType='Data', pcaCount=model_params["pcaCount"])
    fileName_detailedLabels = funcD.getFileName(dataToUse=model_params["dataToUse"], numOfSigns=numOfSigns,
                                                expectedFileType='DetailedLabels', pcaCount=model_params["pcaCount"])
    fileName_labels = funcD.getFileName(dataToUse=model_params["dataToUse"], numOfSigns=numOfSigns,
                                        expectedFileType='Labels', pcaCount=model_params["pcaCount"])

    feat_set = funcD.loadFileIfExist(data_dir, featsFileName)
    detailed_labels_all = funcD.loadFileIfExist(data_dir, fileName_detailedLabels)
    labels_all = funcD.loadFileIfExist(data_dir, fileName_labels)

    return feat_set, labels_all, detailed_labels_all

def displayDataResults(method, dataToUse, posteriorDim, pcaCount, numOfSigns, weightReg = 1.0, batchSize = 16):
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResultFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns,
                                           expectedFileType='BaseResultName', pcaCount=pcaCount)
    baseResultFileNameFull = os.path.join(baseLineResultFolder, baseResultFileName)

    resultDict = np.load(baseResultFileNameFull, allow_pickle=True)
    headerStrFormatBase = "%15s * %10s * %9s "
    headerStrFormat = headerStrFormatBase + "* %6s * %6s * %6s * %6s * %6s"
    #valuesStrFormat = headerStrFormatBase + "* nmiAll(%6.2f) * accAll(%6.2f) * nmiNoz(%6.2f) * accNoz(%6.2f) * emptyClusters(%6d)"
    valuesStrFormat2= headerStrFormatBase + "* %6.2f * %6.2f * %6.2f * %6.2f * %6d"

    print(headerStrFormat % ("npyFileName", "clusModel", "clusCnt", "nmiAll", "accAll", "nmiNoz", "accNoz", "emptyK"))
    baseResults = {}
    baseResultsLab = []
    baseResultsVal = []
    for resultList in resultDict:
        clusterModel = resultList[1]
        clusterCount = resultList[2]
        nmiAll = resultList[3][0]
        accAll = resultList[3][1]
        nmiNoz = resultList[3][2]
        accNoz = resultList[3][3]
        emptyK = resultList[3][4]
        # , '*histCnt=', resultList[3][5][0:10]
        baseResultsLab.append([str(clusterModel)+str(clusterCount)])
        baseResultsVal.append(nmiNoz)
        baseResults[str(clusterModel)+str(clusterCount)] = [nmiNoz]
        print(valuesStrFormat2 %
        (baseResultFileName.replace('.npy', '').replace('_baseResults', ''), clusterModel, clusterCount, nmiAll, accAll, nmiNoz, accNoz, emptyK))

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


#displayDataResults(method='sae', dataToUse='skeleton', posteriorDim=256, pcaCount=32, numOfSigns=11, weightReg = 1.0, batchSize = 16)