import helperFuncs as funcH
import dataLoaderFuncs as funcD
import numpy as np
import os

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

def runClusteringOnFeatSet(data_dir, results_dir, dataToUse, numOfSigns, pcaCount, expectedFileType):
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

    print('running for : ', featsFileName)
    print('featSet(', featSet.shape, '), detailedLabels(', detailedLabels_all.shape, '), labels_All(', labels_all.shape, '), labels_nonzero(', non_zero_labels.shape, ')')

    clustCntVec = [32, 64, 128, 256]
    clusterModelsAll = ['Kmeans', 'GMM_full', 'GMM_diag']
    if os.path.isfile(baseResultFileNameFull):
        print('resultDict will be loaded from(', baseResultFileNameFull, ')')
        resultDict = np.load(baseResultFileNameFull)
        for resultList in resultDict:
            print('clusterModel(', resultList[1], '), clusterCount(', resultList[2], '), resultList=', resultList[3][0:5], ', histCnt=', resultList[3][5][0:10])
    else:
        resultDict = []
        for clusterModel in clusterModelsAll:
            for curClustCnt in clustCntVec:
                print('clusterModel(', clusterModel, '), clusterCount(', curClustCnt, ')')
                predClusters = funcH.clusterData(featVec=featSet, n_clusters=curClustCnt, applyNormalization=False, applyPca=False, clusterModel=clusterModel)

                nmi_cur, acc_cur = funcH.get_NMI_Acc(labels_all, predClusters)

                non_zero_preds = predClusters[np.where(labels_all)]
                nmi_cur_nz, acc_cur_nz = funcH.get_NMI_Acc(non_zero_labels, non_zero_preds)

                numOf_1_sample_bins, histSortedInv = funcH.analyzeClusterDistribution(predClusters, curClustCnt, verbose=2)

                resultList = [featsFileName.replace('.npy', ''), clusterModel, curClustCnt, [nmi_cur, acc_cur, nmi_cur_nz, acc_cur_nz, numOf_1_sample_bins, histSortedInv]]
                resultDict.append(resultList)
                print(resultList[3][0:5], resultList[3][5][0:10])
        np.save(baseResultFileNameFull, resultDict)
    return resultDict


