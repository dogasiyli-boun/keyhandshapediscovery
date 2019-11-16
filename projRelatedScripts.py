import os
import helperFuncs as funcH
import projRelatedHelperFuncs as funcHP

def checkCreateData2Use(data2use, sign_count, recreate=False):
    base_dir = funcH.getVariableByComputerName('base_dir')
    data_dir = funcH.getVariableByComputerName('data_dir')
    nnfolderBase = os.path.join(base_dir, 'neuralNetHandVideos_' + str(sign_count))

    print('signCnt:', sign_count)  # signCnt: 41
    print('nnfolderBase:', nnfolderBase)  # nnfolderBase: / home / dg / DataPath / neuralNetHandVideos_41
    print('exist(nnfolderBase):', os.path.isdir(nnfolderBase))  # exist(nnfolderBase): False

    _ = funcHP.convert_Mat2NPY_sn(data_dir, sign_count, recreate=recreate)
    #/ home / dg / DataPath / bdData / snFeats_41.mat is loaded:)
    #['__globals__', '__header__', '__version__', 'knownKHSlist', 'labelVecs_all', 'surfImArr_all']
    #saving labels((104472,)) at: / home / dg / DataPath / bdData / labels_41.npy
    #saving snFeats((104472, 1600)) at: / home / dg / DataPath / bdData / snFeats_41.npy

    sn_feats_pca = funcHP.createPCAOfData(data_dir, data2use, sign_count, recreate=recreate)
    # loaded sn_feats((104472, 1600)) from: / home / dg / DataPath / bdData / snPCA_41.npy
    # Max of featsPCA = 0.003667559914686907, Min of featsPCA = -0.0028185132292039457

    funcHP.createPCADimsOfData(data_dir, data2use, sign_count, recreate=False)
    # loaded  sn Feats( (104472, 1600) ) from :  /home/dg/DataPath/bdData/snPCA_41.npy
    # Max of featsPCA =  0.003667559914686907 , Min of featsPCA =  -0.0028185132292039457
    # features.shape: (104472, 256)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn256Feats_41.npy
    # features.shape: (104472, 512)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn512Feats_41.npy
    # features.shape: (104472, 1024)
    # saving pca sn features at :  /home/dg/DataPath/bdData/sn1024Feats_41.npy

