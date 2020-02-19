import os
import numpy as np
import helperFuncs as funcH
import projRelatedHelperFuncs as funcHP
import dataLoaderFuncs as funcD

def checkCreateData2Use(sign_count, dataToUse, recreate=False, recreate_hog=False):
    base_dir = funcH.getVariableByComputerName('base_dir')
    data_dir = funcH.getVariableByComputerName('data_dir')
    nnfolderBase = os.path.join(base_dir, 'neuralNetHandVideos_' + str(sign_count))

    print('dataToUse:', dataToUse)  # dataToUse: sn
    print('signCnt:', sign_count)  # signCnt: 41
    print('nnfolderBase:', nnfolderBase)  # nnfolderBase: / home / dg / DataPath / neuralNetHandVideos_41
    print('exist(nnfolderBase):', os.path.isdir(nnfolderBase))  # exist(nnfolderBase): False
    if dataToUse == 'sn':
        featureStr = 'surfImArr_all'
        labelStr = 'labelVecs_all'
        possible_fname_init = ['surfImArr', 'snFeats']
        dimArray = [256, 512, 1024]
        convertMat2NPY = True
    elif dataToUse == 'skeleton':
        featureStr = 'skel_all'
        labelStr = 'labelVecs_all'
        possible_fname_init = ['skeleton', 'skelFeats']
        dimArray = [32, 64, 96]
        convertMat2NPY = True
    elif dataToUse == 'hog':
        dimArray = [256, 512, 1024]
        convertMat2NPY = False
    elif dataToUse == 'hgsnsk' or dataToUse == 'hgsn' or dataToUse == 'hgsk' or dataToUse == 'snsk':
        dimArray = [256, 512, 1024]
        convertMat2NPY = False
    else:
        os.exit(5)

    if convertMat2NPY:
        _ = funcHP.convert_Mat2NPY(dataToUse, data_dir, sign_count, featureStr, labelStr, possible_fname_init, recreate=recreate)
    elif dataToUse == 'hog':
        _ = funcD.loadData_hog(loadHogIfExist=not recreate_hog, numOfSigns=sign_count)
    # /home/dg/DataPath/bdData/snFeats_41.mat is loaded:)
    # ['__globals__', '__header__', '__version__', 'knownKHSlist', 'labelVecs_all', 'surfImArr_all']
    # saving labels((104472,)) at: / home / dg / DataPath / bdData / labels_41.npy
    # saving snFeats((104472, 1600)) at: / home / dg / DataPath / bdData / snFeats_41.npy

    for normMode in ['']:
        _ = funcHP.createPCAOfData(data_dir, dataToUse, sign_count, recreate=recreate, normMode=normMode)
    # loaded sn_feats((104472, 1600)) from: / home / dg / DataPath / bdData / snPCA_41.npy
    # Max of featsPCA = 0.003667559914686907, Min of featsPCA = -0.0028185132292039457

    for normMode in ['']:
        funcHP.createPCADimsOfData(data_dir, dataToUse, sign_count, dimArray, recreate=recreate, normMode=normMode)
    # loaded  sn Feats( (104472, 1600) ) from :  /home/dg/DataPath/bdData/snPCA_41.npy
    # Max of featsPCA =  0.003667559914686907 , Min of featsPCA =  -0.0028185132292039457
    # features.shape: (104472, 256)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn256Feats_41.npy
    # features.shape: (104472, 512)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn512Feats_41.npy
    # features.shape: (104472, 1024)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn1024Feats_41.npy

    for normMode in ['']:
        for dims in dimArray:
            funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, featType=dataToUse, numOfSigns=sign_count, pcaCount=dims, expectedFileType='Data', normMode=normMode)
    funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, featType=dataToUse, numOfSigns=sign_count, pcaCount=-1, expectedFileType='Data', normMode='')

def run4All_createData(dataToUseArr=["hog", "skeleton", "sn"], sign_countArr=[11]):
    for dataToUse in dataToUseArr:
        for sign_count in sign_countArr:
            checkCreateData2Use(sign_count=sign_count, dataToUse=dataToUse, recreate=False, recreate_hog=False)

def runForBaseClusterResults(normMode, randomSeed=5, clusterModels=['KMeans', 'GMM_diag'],
                             dataToUseArr=["hog", "skeleton", "sn"], numOfSignsArr=[11], clustCntVec=None):
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName('results_dir')

    for dataToUse in dataToUseArr:
        for numOfSigns in numOfSignsArr:
            if clustCntVec is None:
                clustCntVec = [64, 128, 256]
            if dataToUse == 'skeleton':
                dimArray = [32, 64, 96]
            else:  # dataToUse == 'sn' or dataToUse == 'hog':
                dimArray = [256]
            for dims in dimArray:
                funcHP.runClusteringOnFeatSet(data_dir=data_dir, results_dir=results_dir, dataToUse=dataToUse,
                                              normMode=normMode, numOfSigns=numOfSigns, pcaCount=dims,
                                              expectedFileType='Data', clustCntVec=clustCntVec, clusterModels=clusterModels,
                                              randomSeed=randomSeed)

#
def runForBaseClusterResults_OPTICS(randomSeed = 5, clustCntVec = [32, 64, 128, 256, 512],
                                    dataToUseArr=["hog", "skeleton", "sn"], numOfSignsVec = [11, 41]):
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName('results_dir')
    for dataToUse in dataToUseArr:
        for numOfSigns in numOfSignsVec:
            if dataToUse == 'skeleton':
                dimArray = [96]  # 32, 64,
            else:  # dataToUse == 'sn' or dataToUse == 'hog':
                dimArray = [256]  # 512
            for dims in dimArray:
                funcHP.runOPTICSClusteringOnFeatSet(data_dir=data_dir, results_dir=results_dir, dataToUse=dataToUse,
                                              numOfSigns=numOfSigns, pcaCount=dims, expectedFileType='Data',
                                              clustCntVec=clustCntVec, randomSeed=randomSeed)

def createCombinedDatasets(numOfSigns = 11):
    data_dir = funcH.getVariableByComputerName('data_dir')
    fName_sn = funcD.getFileName('sn', '', -1, numOfSigns, 'Data')
    fName_hg = funcD.getFileName('hog', '', -1, numOfSigns, 'Data')
    fName_sk = funcD.getFileName('skeleton', '', -1, numOfSigns, 'Data')

    X_sn = funcD.loadFileIfExist(directoryOfFile=data_dir, fileName=fName_sn)
    X_hg = funcD.loadFileIfExist(directoryOfFile=data_dir, fileName=fName_hg)
    X_sk = funcD.loadFileIfExist(directoryOfFile=data_dir, fileName=fName_sk)

    print('X_sn.shape = ', X_sn.shape)
    print('X_hg.shape = ', X_hg.shape)
    print('X_sk.shape = ', X_sk.shape)

    X_hgsnsk = np.concatenate([X_hg.T, X_sn.T, X_sk.T]).T
    X_hgsn = np.concatenate([X_hg.T, X_sn.T]).T
    X_hgsk = np.concatenate([X_hg.T, X_sk.T]).T
    X_snsk = np.concatenate([X_sn.T, X_sk.T]).T

    print('X_hgsnsk.shape = ', X_hgsnsk.shape)
    print('X_hgsn.shape = ', X_hgsn.shape)
    print('X_hgsk.shape = ', X_hgsk.shape)
    print('X_snsk.shape = ', X_snsk.shape)

    fName_hgsnsk = os.path.join(data_dir, fName_hg.replace("hog", "hgsnsk"))
    fName_hgsn = os.path.join(data_dir, fName_hg.replace("hog", "hgsn"))
    fName_hgsk = os.path.join(data_dir, fName_hg.replace("hog", "hgsk"))
    fName_snsk = os.path.join(data_dir, fName_hg.replace("hog", "snsk"))

    if os.path.isfile(os.path.join(data_dir, fName_hgsnsk)):
        _ = np.load(fName_hgsnsk)
    else:
        np.save(fName_hgsnsk, X_hgsnsk)

    if os.path.isfile(os.path.join(data_dir, fName_hgsn)):
        _ = np.load(fName_hgsn)
    else:
        np.save(fName_hgsn, X_hgsn)

    if os.path.isfile(os.path.join(data_dir, fName_hgsk)):
        _ = np.load(fName_hgsk)
    else:
        np.save(fName_hgsk, X_hgsk)

    if os.path.isfile(os.path.join(data_dir, fName_snsk)):
        _ = np.load(fName_snsk)
    else:
        np.save(fName_snsk, X_snsk)



#  resultDict = funcHP.runClusteringOnFeatSet(data_dir=funcH.getVariableByComputerName('data_dir'),
#                                           results_dir=funcH.getVariableByComputerName('results_dir'),
#                                           dataToUse='skeleton', numOfSigns=11, pcaCount=32,
#                                           expectedFileType='Data', clusterModels=['KMeans', 'GMM_diag'], randomSeed=5)
#  runForBaseClusterResults(normMode='', clusterModels = ['KMeans', 'GMM_diag'])
#  runForBaseClusterResults_OPTICS(randomSeed = 5, clustCntVec = [32, 64])
#  run4All_createData(sign_countArr=[12])
#  createCombinedDatasets(numOfSigns = 41)
#  checkCreateData2Use(41, "snsk", recreate=False, recreate_hog=False)