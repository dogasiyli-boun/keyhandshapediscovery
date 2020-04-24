import helperFuncs as funcH
import dataLoaderFuncs as funcD
import ensembleFuncs as funcEns
import visualize as funcVis
import numpy as np
import os
from numpy.random import seed
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import time
import datetime
from sklearn.metrics import confusion_matrix

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
                predClusters = funcH.clusterData(featVec=featSet, n_clusters=curClustCnt, normMode='', applyPca=False, clusterModel=clusterModel)
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
                predClusters = funcH.clusterData(featVec=featSet, n_clusters=curClustCnt, norMMode='', applyPca=False, clusterModel=clusterModel)
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
                                                numOfSigns=numOfSigns,expectedFileType='DetailedLabels')
    fileName_labels = funcD.getFileName(dataToUse=model_params["dataToUse"], normMode=str(model_params["normMode"]), pcaCount=model_params["pcaCount"],
                                        numOfSigns=numOfSigns, expectedFileType='Labels')

    feat_set = funcD.loadFileIfExist(data_dir, featsFileName)
    if feat_set.size == 0:
        _ = funcH.createPCAOfData(data_dir, dataToUse=model_params["dataToUse"], sign_count=numOfSigns, recreate=False, normMode=str(model_params["normMode"]))
        feat_set = funcD.loadFileIfExist(data_dir, featsFileName)
        if feat_set.size == 0:
            os.error("finish here")
        # apply needed transofrmation and use the data


    detailed_labels_all = funcD.loadFileIfExist(data_dir, fileName_detailedLabels)
    labels_all = funcD.loadFileIfExist(data_dir, fileName_labels)

    return feat_set, labels_all, detailed_labels_all

def loadBaseResult(fileName):
    results_dir = funcH.getVariableByComputerName('results_dir')
    preds = np.load(os.path.join(results_dir, 'baseResults', fileName + '.npz'))
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
        _confMat, kluster2Classes = funcH.countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=labelNames)

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


def ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                     consensus_clustering_max_k=256, useNZ=True, nos=11,
                     resultsToCombineDescriptorStr="",
                     labelNames = None, verbose=False):
    N = cluster_runs.shape[0]

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
        klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels, predictionsDict[i]["prd"], labelNames,
                                                                predictionsDict[i]["str"])
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
    print("\r\nclassification metrics comparison\r\n")
    print(classRet)
    print("\r\n")

    c_pdf = resultsDict[0]["c_pdf"][['class', '%f1']].sort_index().rename(
        columns={"class": "f1Score", "%f1": predictionsDict[0]["str"]})
    for i in range(1, N + 1):
        c_pdf.insert(i + 1, predictionsDict[i]["str"], resultsDict[i]["c_pdf"][['%f1']].sort_index(), True)
    print("\r\nf1 score comparisons for classes\r\n")
    print(c_pdf)
    print("\r\n")

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

    results_dir = funcH.getVariableByComputerName("results_dir")
    predictResultFold = os.path.join(results_dir, "predictionResults")

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
                purity_k, correctLabelInds, mappedClass = funcH.calcPurity(list(sampleLabelsPicked))
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

    #displayDataResults(method='sae', dataToUse='skeleton', posteriorDim=256, pcaCount=32, numOfSigns=11, weightReg = 1.0, batchSize = 16)

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
                max_val = np.max(featsMat["test"].values[:29])
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