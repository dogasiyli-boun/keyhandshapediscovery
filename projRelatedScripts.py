import os
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
    else:
        os.exit(5)

    if convertMat2NPY:
        _ = funcHP.convert_Mat2NPY(dataToUse, data_dir, sign_count, featureStr, labelStr, possible_fname_init, recreate=recreate)
    else:
        _ = funcD.loadData_hog(loadHogIfExist=not recreate_hog, numOfSigns=sign_count)
    # /home/dg/DataPath/bdData/snFeats_41.mat is loaded:)
    # ['__globals__', '__header__', '__version__', 'knownKHSlist', 'labelVecs_all', 'surfImArr_all']
    # saving labels((104472,)) at: / home / dg / DataPath / bdData / labels_41.npy
    # saving snFeats((104472, 1600)) at: / home / dg / DataPath / bdData / snFeats_41.npy

    _ = funcHP.createPCAOfData(data_dir, dataToUse, sign_count, recreate=recreate)
    # loaded sn_feats((104472, 1600)) from: / home / dg / DataPath / bdData / snPCA_41.npy
    # Max of featsPCA = 0.003667559914686907, Min of featsPCA = -0.0028185132292039457

    funcHP.createPCADimsOfData(data_dir, dataToUse, sign_count, dimArray, recreate=recreate)
    # loaded  sn Feats( (104472, 1600) ) from :  /home/dg/DataPath/bdData/snPCA_41.npy
    # Max of featsPCA =  0.003667559914686907 , Min of featsPCA =  -0.0028185132292039457
    # features.shape: (104472, 256)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn256Feats_41.npy
    # features.shape: (104472, 512)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn512Feats_41.npy
    # features.shape: (104472, 1024)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn1024Feats_41.npy

    for dims in dimArray:
        funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, featType=dataToUse, numOfSigns=sign_count, pcaCount=dims, expectedFileType='Data')
    funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, featType=dataToUse, numOfSigns=sign_count, pcaCount=-1, expectedFileType='Data')

def run4All_createData():
    for dataToUse in ["hog", "skeleton", "sn"]:
        for sign_count in [11, 41]:
            checkCreateData2Use(sign_count=sign_count, dataToUse=dataToUse, recreate=False, recreate_hog=False)

def runForBaseClusterResults(randomSeed = 5, clusterModels = ['Kmeans', 'GMM_diag', 'GMM_full', 'Spectral']):
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName('results_dir')

    for dataToUse in ["hog", "skeleton", "sn"]:
        for numOfSigns in [11, 41]:
            if dataToUse == 'sn' or dataToUse == 'hog':
                dimArray = [-1, 256, 512, 1024]
            elif dataToUse == 'skeleton':
                dimArray = [-1, 32, 64, 96]
            for dims in dimArray:
                funcHP.runClusteringOnFeatSet(data_dir=data_dir, results_dir=results_dir, dataToUse=dataToUse,
                                              numOfSigns=numOfSigns, pcaCount=dims, expectedFileType='Data',
                                              clusterModels=clusterModels, randomSeed=randomSeed)

#resultDict = funcHP.runClusteringOnFeatSet(data_dir=funcH.getVariableByComputerName('data_dir'),
#                                           results_dir=funcH.getVariableByComputerName('results_dir'),
#                                           dataToUse='skeleton', numOfSigns=11, pcaCount=32,
#                                           expectedFileType='Data', clusterModels=['Kmeans', 'GMM_diag'], randomSeed=5)
#runForBaseClusterResults(randomSeed = 5)