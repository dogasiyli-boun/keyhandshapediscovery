import helperFuncs as funcH
import numpy as np
import os

#features, labels = getFeatsFromMat(mat,'dataCurdim', 'labelVecs_all')
def getFeatsFromMat(mat, featureStr, labelStr):
    features = np.asarray(mat[featureStr], dtype=float)
    labels = np.asarray(mat[labelStr], dtype=int)
    return features, labels

def getSurfNormFile(data_dir, signCnt):
    matFileName_1 = os.path.join(data_dir, 'surfImArr_' + str(signCnt) + '.mat')
    matFileName_2 = os.path.join(data_dir, 'snFeats_' + str(signCnt) + '.mat')
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

def createPCAOfData(data_dir, dataToUse, signCnt, recreate=False):
    npy_PCAFileName = os.path.join(data_dir,  dataToUse + 'PCA_' + str(signCnt) + '.npy')
    if os.path.isfile(npy_PCAFileName) and not recreate:
        featsPCA = np.load(npy_PCAFileName)
        print('loaded ', dataToUse, 'Feats(', featsPCA.shape, ') from : ', npy_PCAFileName)
        print('Max of featsPCA = ', np.amax(featsPCA) ,', Min of featsPCA = ', np.amin(featsPCA))
    else:
        npy_FeatsFileName = os.path.join(data_dir,  dataToUse + 'Feats_' + str(signCnt) + '.npy')
        feats = np.load(npy_FeatsFileName)
        print('Max of feats = ', np.amax(feats), ', Min of feats = ', np.amin(feats))
        feats_pca = funcH.applyMatTransform(feats, applyNormalization=True, applyPca=True, whiten=True)
        print('Max of featsPCA = ', np.amax(feats_pca), ', Min of featsPCA = ', np.amin(feats_pca))
        np.save(npy_PCAFileName, feats_pca)
    return featsPCA

def convert_Mat2NPY_sn(data_dir, signCnt, recreate=False):
    npy_labels_file_name = os.path.join(data_dir, 'labels_' + str(signCnt) + '.npy')
    npy_sn_feats_file_name = os.path.join(data_dir, 'snFeats_' + str(signCnt) + '.npy')

    if not recreate and os.path.isfile(npy_labels_file_name) and os.path.isfile(npy_sn_feats_file_name):
        labels = np.load(npy_labels_file_name)
        print('loaded labels(', labels.shape, ') from : ', npy_labels_file_name)
        sn_feats = np.load(npy_sn_feats_file_name)
        print('loaded snFeats(', sn_feats.shape, ') from : ', npy_sn_feats_file_name)
    else:
        mat_sn_feats = getSurfNormFile(data_dir, signCnt)
        sn_feats, labels = getFeatsFromMat(mat_sn_feats, 'surfImArr_all', 'labelVecs_all')

        labels = np.reshape(labels, (len(labels), -1)).squeeze()
        print('saving labels(', labels.shape, ') at : ', npy_labels_file_name)
        np.save(npy_labels_file_name, labels)
        print('saving snFeats(', sn_feats.shape, ') at : ', npy_sn_feats_file_name)
        np.save(npy_sn_feats_file_name, sn_feats)
    return sn_feats, labels

def createPCADimsOfData(data_dir, data2use, sign_count, dimArray = [256, 512, 1024], recreate=False):
    featsPCA = createPCAOfData(data_dir, data2use, recreate=recreate)
    for dims in dimArray:
        npy_PCAFileName = os.path.join(data_dir,  data2use + str(dims) + 'Feats_' + str(sign_count) + '.npy')
        featsToSave = featsPCA[:,0:dims]
        if os.path.isfile(npy_PCAFileName) and not recreate:
            featsToSave = np.load(npy_PCAFileName)
            print('pca exists at : ', npy_PCAFileName)
        else:
            print("features.shape:", featsToSave.shape)
            print('saving pca sn features at : ', npy_PCAFileName)
            np.save(npy_PCAFileName, featsToSave)

