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
                    '_cRM' + str(trainParams["corr_randMode"])
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
                    '_cRM' + str(trainParams["corr_randMode"])
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
#displayDataResults(method='sae', dataToUse='skeleton', posteriorDim=256, pcaCount=32, numOfSigns=11, weightReg = 1.0, batchSize = 16)