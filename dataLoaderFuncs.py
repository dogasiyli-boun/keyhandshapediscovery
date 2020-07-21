import os
import numpy as np
from skimage import data
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import glob
import helperFuncs as funcH
import pandas as pd
from torch.utils.data import Dataset, random_split

def loadFileIfExist(directoryOfFile, fileName):
    fileNameFull = directoryOfFile + os.sep + fileName
    if os.path.isfile(fileNameFull):
        X = np.load(fileNameFull)
    else:
        X = np.array([])
    return X

def getFileName(dataToUse, normMode, pcaCount, numOfSigns, expectedFileType):
    pcaCountStr = str(pcaCount) if pcaCount > 0 else ""
    normModeStr = normMode if normMode == "" else "_" + normMode + "_"
    if expectedFileType == 'PCA':
        fileName_PCA            = dataToUse + normModeStr + 'PCA' + '_' + str(numOfSigns) + '.npy'  # 'hogPCA_41.npy' or 'hogPCA_11.npy' or 'skeletonFeats_11.npy' or 'snFeats_11.npy'
        fileName = fileName_PCA
    if expectedFileType == 'Data':
        fileName_Data           = dataToUse + normModeStr + pcaCountStr + 'Feats' + '_' + str(numOfSigns) + '.npy'  # 'hogFeats_41.npy' or 'hogFeats_11.npy' or 'skeletonFeats_11.npy' or 'snFeats_11.npy'
        fileName = fileName_Data
    if expectedFileType == 'Labels':
        fileName_Labels         = 'labels' + '_' + str(numOfSigns) + '.npy'  # 'labels_41.npy' or 'labels_11.npy'
        fileName = fileName_Labels
    if expectedFileType == 'DetailedLabels':
        fileName_DetailedLabels = 'detailedLabels' + '_' + str(numOfSigns) + '.npy'  # 'detailedLabels_41.npy' or 'detailedLabels_11.npy'
        fileName = fileName_DetailedLabels
    if expectedFileType == 'CorrespendenceVec':
        fileName_Corr = dataToUse + normModeStr + pcaCountStr + '_corrFrames' + '_' + str(numOfSigns) + '.npy'  # 'hog_corrFrames_41.npy' or 'hog_corrFrames_11.npy'
        fileName = fileName_Corr
    if expectedFileType == 'BaseResultName':
        fileName_BaseRes = dataToUse + normModeStr + pcaCountStr + '_' + str(numOfSigns) + '_baseResults' + '.npy'  # 'hog_baseResults_41.npy' or 'hog_baseResults_11.npy'
        fileName = fileName_BaseRes
    return fileName

def getFileNameGroup(dataToUse, normMode, pcaCount, numOfSigns, expectedFileType, data_dir=''):
    featsFileName = os.path.join(data_dir, getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType=expectedFileType)) # 'hogFeats_41.npy', 'skeletonFeats_41.npy'
    labelsFileName = os.path.join(data_dir, getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='Labels')) # 'detailedLabels_41.npy'
    detailedLabelsFileName = os.path.join(data_dir, getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='DetailedLabels')) # 'detailedLabels_41.npy'
    corrFramesFileName = os.path.join(data_dir, getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='CorrespendenceVec'))
    return featsFileName, labelsFileName, detailedLabelsFileName, corrFramesFileName

def loadSkeletonDataFromVideosFolders(base_dir=funcH.getVariableByComputerName('base_dir'),
                                      data_dir=funcH.getVariableByComputerName('data_dir'),
                                      loadIfExist=True, numOfSigns=11):

    videosFolderName = 'neuralNetHandVideos_' + str(numOfSigns)
    base_dir_train_feat = os.path.join(base_dir, videosFolderName)

    featsFileNameFull, labelsFileNameFull, detailedLabelsFileNameFull, _ = getFileNameGroup(dataToUse='skeleton', normMode='', pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='Data', data_dir=data_dir)
    # 'skeletonFeats_41.npy' or 'skeletonFeats_11.npy'

    if loadIfExist and os.path.isfile(featsFileNameFull) and os.path.isfile(labelsFileNameFull) and os.path.isfile(detailedLabelsFileNameFull):
        print('loading exported feat_set from(', featsFileNameFull, ')')
        feat_set = np.load(featsFileNameFull)
        labels_all = np.load(labelsFileNameFull)
        detailedLabels_all = np.load(detailedLabelsFileNameFull)
        print('loaded exported feat_set(', feat_set.shape, ') from(', featsFileNameFull, ')')
    else:
        detailedLabels_all = np.array([0, 0, 0, 0])
        labels_all = np.array([0, 0, 0, 0])
        feat_set = np.array([0, 0, 0, 0])
        foldernames = np.sort(os.listdir(base_dir_train_feat))
        signID = 0
        frameCount = 0
        for f in foldernames:
            sign_folder = os.path.join(base_dir_train_feat, str(f).format(':02d'))
            if not os.path.isdir(sign_folder):
                continue
            signID = signID + 1
            videoID = 0
            videos = np.sort(os.listdir(sign_folder))
            print(f)
            print('going to create hog from sign folder(', sign_folder, ')')
            for v in videos:
                video_folder = os.path.join(sign_folder, v)
                if not os.path.isdir(video_folder):
                    continue
                videoID = videoID + 1
                print('going to create hog from video folder(', video_folder, ')')
                frames = os.listdir(video_folder)
                feat_set_video = np.array([0, 0, 0, 0])

                skelFeat_file = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('skel.txt')]
                featsMat = pd.read_csv(skelFeat_file[0], header=None)
                frameCntSkel = featsMat.shape[0]

                labels_file = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('labels.txt')]
                labels = np.loadtxt(os.path.join(video_folder, labels_file[0]))
                frameCntLabel = len(labels)

                frameList = video_folder + os.sep + '*.png'
                pngCount = len(glob.glob(frameList))

                assert (pngCount==frameCntLabel & pngCount==frameCntSkel), \
                    "these three values must be same frameCntSkel(" + str(frameCntSkel) + ")" + \
                                                    "frameCntLabel(" + str(frameCntLabel) + ")" + \
                                                    "pngCount(" + str(pngCount) + ")"


                fr = frameCount
                to = frameCount + frameCntLabel
                frCnt = to - fr
                frameIDs = np.asarray(range(fr, to)).reshape(frCnt, -1)
                detailedLabels_video = np.hstack((signID * np.ones([frCnt, 1]), videoID * np.ones([frCnt, 1]), frameIDs, np.asarray(labels).reshape(frCnt, -1)))

                if np.all(feat_set == 0):
                    feat_set = featsMat
                else:
                    feat_set = np.vstack((feat_set, featsMat))

                if np.all(labels_all == 0):
                    labels_all = labels
                else:
                    labels_all = np.hstack((labels_all, labels))

                if np.all(detailedLabels_all == 0):
                    detailedLabels_all = detailedLabels_video
                else:
                    detailedLabels_all = np.vstack((detailedLabels_all, detailedLabels_video))
                frameCount = len(labels_all)

        print('saving exported feat_set(', feat_set.shape, ') into(', featsFileNameFull, ')')
        np.save(featsFileNameFull, feat_set)
        print('saving labels(', labels_all.shape, ') into(', labelsFileNameFull, ')')
        np.save(labelsFileNameFull, labels_all)
        print('saving detailedLabels(', detailedLabels_all.shape, ') into(', detailedLabelsFileNameFull, ')')
        np.save(detailedLabelsFileNameFull, detailedLabels_all)

    return feat_set, labels_all, detailedLabels_all

def loadData_hog(base_dir = funcH.getVariableByComputerName('base_dir'), data_dir = funcH.getVariableByComputerName('data_dir'),
                 loadHogIfExist=True, numOfSigns=11):

    videosFolderName = 'neuralNetHandVideos_' + str(numOfSigns)
    base_dir_train_feat = os.path.join(base_dir, videosFolderName)

    hogFeatsFileName = getFileName(dataToUse='hog', normMode='', pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='Data')
    hogFeatsFileNameFull = os.path.join(data_dir, hogFeatsFileName)
    labelsFileNameFull = os.path.join(data_dir, getFileName(dataToUse='hog', normMode='', pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='Labels'))
    detailedLabelsFileNameFull = os.path.join(data_dir, getFileName(dataToUse='hog', normMode='', pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='DetailedLabels'))

    if loadHogIfExist and os.path.isfile(hogFeatsFileNameFull) and os.path.isfile(labelsFileNameFull) and os.path.isfile(detailedLabelsFileNameFull):
        print('loading exported feat_set from(', hogFeatsFileNameFull, ')')
        feat_set = np.load(hogFeatsFileNameFull)
        labels_all = np.load(labelsFileNameFull)
        detailedLabels_all = np.load(detailedLabelsFileNameFull)
        print('loaded exported feat_set(', feat_set.shape, ') from(', hogFeatsFileName, ')')
    else:
        detailedLabels_all = np.array([0, 0, 0, 0])
        labels_all = np.array([0, 0, 0, 0])
        feat_set = np.array([0, 0, 0, 0])
        foldernames = np.sort(os.listdir(base_dir_train_feat))
        signID = 0
        frameCount = 0
        for f in foldernames:
            sign_folder = os.path.join(base_dir_train_feat, str(f).format(':02d'))
            if not os.path.isdir(sign_folder):
                continue
            signID = signID + 1
            videoID = 0
            videos = np.sort(os.listdir(sign_folder))
            print(f)
            print('going to create hog from sign folder(', sign_folder, ')')
            for v in videos:
                video_folder = os.path.join(sign_folder, v)
                if not os.path.isdir(video_folder):
                    continue
                videoID = videoID + 1
                print('going to create hog from video folder(', video_folder, ')')
                frames = os.listdir(video_folder)
                feat_set_video = np.array([0, 0, 0, 0])

                olderFileName_v01 = video_folder + os.sep + 'hog_set.npz'
                olderFileName_v02  = video_folder + os.sep + 'hog_set_41.npz'
                hogFeats_curVideo_FileNameFull = video_folder + os.sep + hogFeatsFileName.replace('.npy', '.npz')
                if os.path.isfile(olderFileName_v01):
                    os.rename(olderFileName_v01, hogFeats_curVideo_FileNameFull)
                elif os.path.isfile(olderFileName_v02):
                    os.rename(olderFileName_v02, hogFeats_curVideo_FileNameFull)

                feats_labels_loaded = False
                if os.path.isfile(hogFeats_curVideo_FileNameFull):
                    npzfile = np.load(hogFeats_curVideo_FileNameFull)
                    feat_set_video = npzfile['feat_set_video']
                    labels = npzfile['labels']
                    feats_labels_loaded = True

                if feats_labels_loaded:
                    frameList = video_folder + os.sep + '*.png'
                    pngCount = len(glob.glob(frameList))
                    feats_labels_loaded = pngCount==len(labels)

                if not feats_labels_loaded:
                    for frame in sorted(frames):
                        if frame.endswith('.png'):
                            frame_name = os.path.join(video_folder, frame)
                            img = data.load(frame_name)
                            feat_current = hog(img, pixels_per_cell=(32, 32), cells_per_block=(4, 4))
                            if np.all(feat_set_video == 0):
                                feat_set_video = feat_current
                            else:
                                feat_set_video = np.vstack((feat_set_video, feat_current))
                        elif frame.endswith('_labels.txt'):
                            labels = np.loadtxt(os.path.join(video_folder, frame))
                    np.savez(hogFeats_curVideo_FileNameFull, feat_set_video=feat_set_video, labels=labels)

                fr = frameCount
                to = frameCount + len(labels)
                frCnt = to - fr
                frameIDs = np.asarray(range(fr, to)).reshape(frCnt, -1)
                detailedLabels_video = np.hstack((signID * np.ones([frCnt, 1]), videoID * np.ones([frCnt, 1]), frameIDs, np.asarray(labels).reshape(frCnt, -1)))

                if np.all(feat_set == 0):
                    feat_set = feat_set_video
                else:
                    feat_set = np.vstack((feat_set, feat_set_video))

                if np.all(labels_all == 0):
                    labels_all = labels
                else:
                    labels_all = np.hstack((labels_all, labels))

                if np.all(detailedLabels_all == 0):
                    detailedLabels_all = detailedLabels_video
                else:
                    detailedLabels_all = np.vstack((detailedLabels_all, detailedLabels_video))
                frameCount = len(labels_all)
        print('saving exported feat_set(', feat_set.shape, ') into(', hogFeatsFileNameFull, ')')

        np.save(hogFeatsFileNameFull, feat_set)
        np.save(labelsFileNameFull, labels_all)
        np.save(detailedLabelsFileNameFull, detailedLabels_all)

    return feat_set, labels_all, detailedLabels_all

def getCorrespondentFrames(base_dir, data_dir, featType, normMode, pcaCount, numOfSigns=11, expectedFileType='Data', verbose=1):
    videosFolderName = 'neuralNetHandVideos_' + str(numOfSigns)
    featsFileName, _, detailedLabelsFileName, corrFramesFileName = getFileNameGroup(dataToUse=featType, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType=expectedFileType, data_dir=data_dir)
    corrFramesSignFileName_Base = str.replace(getFileName(dataToUse=featType, normMode=normMode, pcaCount=pcaCount, numOfSigns=numOfSigns, expectedFileType='CorrespendenceVec'), ".npy", "SSS.npy")

    base_dir_nn = os.path.join(base_dir, videosFolderName)
    detailedLabelsFileNameFull = detailedLabelsFileName
    detailedLabels_all = np.load(detailedLabelsFileNameFull)

    featSet = np.load(featsFileName)

    if verbose > 0:
        print('within getCorrespondentFrames func: featSet.shape(', featSet.shape, '), detailedLabels_all.shape(', detailedLabels_all.shape, ')')

    if os.path.isfile(corrFramesFileName):
        corrFramesAll = np.load(corrFramesFileName)
        if verbose > 0:
            print('loaded corrFramesAll of shape ', corrFramesAll.shape, ' from(', corrFramesFileName, ')')
    else:
        signNames = np.sort(os.listdir(base_dir_nn))
        signID = 0
        corrFramesAll = np.array([])
        for signCur in signNames:
            corrFramesSign = np.array([])
            sign_folder = os.path.join(base_dir_nn, signCur)
            if not os.path.isdir(sign_folder):
                continue
            signID = signID + 1
            videos = [f for f in os.listdir(sign_folder) if os.path.isdir(os.path.join(sign_folder, f))]
            videos = np.sort(videos)
            vidCnt = len(videos)
            detailedLabels_all_sign_rows = np.argwhere(detailedLabels_all[:, 0] == signID).flatten()

            corrFramesSignFileName = os.path.join(sign_folder, corrFramesSignFileName_Base.replace("SSS", "_s" + str(signCur)))

            if os.path.isfile(corrFramesSignFileName):
                corrFramesSign = np.load(corrFramesSignFileName)
                print('loaded corrFramesSign of shape ', corrFramesSign.shape)
            else:
                for v1 in range(0, vidCnt):
                    if verbose > 1:
                        print('s_', signCur, ', v(', videos[v1], ') to vCnt - ', vidCnt-v1)
                    for v2 in range(v1 + 1, vidCnt):
                        detailedLabels_cur_vid_rows_rel = np.argwhere(detailedLabels_all[detailedLabels_all_sign_rows, 1] == v1+1).flatten()
                        frIDs_v1 = detailedLabels_all_sign_rows[detailedLabels_cur_vid_rows_rel]
                        feat_set_v1 = featSet[frIDs_v1, :]

                        detailedLabels_cur_vid_rows_rel = np.argwhere(detailedLabels_all[detailedLabels_all_sign_rows, 1] == v2+1).flatten()
                        frIDs_v2 = detailedLabels_all_sign_rows[detailedLabels_cur_vid_rows_rel]
                        feat_set_v2 = featSet[frIDs_v2, :]

                        if feat_set_v1.shape[0]!=frIDs_v1.shape[0] or feat_set_v2.shape[0]!=frIDs_v2.shape[0]:
                            print('s_', signCur, ', v ', videos[v1], ' to vCnt-', vidCnt - v1)
                            print("feat_set_v1.shape(", feat_set_v1.shape, "), frIDs_v1.shape(", frIDs_v1.shape, ")")
                            print("feat_set_v2.shape(", feat_set_v2.shape, "), frIDs_v2.shape(", frIDs_v2.shape, ")")
                            os._exit(3)

                        if verbose > 1:
                            print('s_', signCur, ', v1(', str(np.min(frIDs_v1)), ' to ', str(np.max(frIDs_v1)), ') and v2(', str(np.min(frIDs_v2)), ' to ', str(np.max(frIDs_v2)), ')')

                        corrPath = funcH.getCorrPath(feat_set_v1, feat_set_v2, frIDs_v1, frIDs_v2, metric='euclidean')
                        if corrFramesSign.size == 0:
                            corrFramesSign = corrPath
                        else:
                            corrFramesSign = np.hstack((corrFramesSign, corrPath))
                        #print('s_', signCur, ', v(', videos[v1], '-', frCnt_1, ') vs v(', videos[v2], '-', frCnt_2, ')')
                        # print(corrPath)
                        # I need to load the features for v1 and v2 here
                        # and get dtw of them
                        # then assign for all frames their correspondant frame from the other video
                        # featSet_v1 = np.random.rand(14,50)
                        # featSet_v2 = np.random.rand(22,50)
                np.save(corrFramesSignFileName, corrFramesSign)
                if verbose > 0:
                    print('saved corrFramesSign(', corrFramesSignFileName, ') of shape ', corrFramesSign.shape)
            if corrFramesAll.size == 0:
                corrFramesAll = corrFramesSign
            else:
                corrFramesAll = np.hstack((corrFramesAll, corrFramesSign))
        np.save(corrFramesFileName, corrFramesAll)
        if verbose > 0:
            print('saved corrFrames(', corrFramesFileName, ') of shape ', corrFramesAll.shape)
    return corrFramesAll, detailedLabels_all

def applyPCA2Data(feat_set, data_dir, data_dim, dataToUse, normMode, numOfSigns=11, loadIfExist = True):
    # normMode = str(modelParams["normMode"])
    pcaFeatsFileName = getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='PCA')
    hogFeatsFileName = getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='Data')

    pcaFeatsFileNameFull = os.path.join(data_dir, pcaFeatsFileName)
    inputFeatsFileNameFull = os.path.join(data_dir, hogFeatsFileName)

    if loadIfExist and os.path.isfile(pcaFeatsFileNameFull):
        print('loading feat_set_pca from(', pcaFeatsFileNameFull, ')')
        feat_set_pca = np.load(pcaFeatsFileNameFull)
        print('loaded feat_set_pca(',  feat_set_pca.shape, ') from(', pcaFeatsFileName, ')')
        return feat_set_pca

    if os.path.isfile(inputFeatsFileNameFull) and feat_set.size == 0:
        feat_set = np.load(inputFeatsFileNameFull)

    pca = PCA(n_components=data_dim)
    feat_set_pca = pca.fit_transform(feat_set)
    np.save(pcaFeatsFileNameFull, feat_set_pca)

    return feat_set_pca

def loadPCAData(dataToUse, normMode, numOfSigns, data_dim, skipLoadOfOriginalData, data_dir, base_dir = funcH.getVariableByComputerName('base_dir')):
    # normMode = str(modelParams["normMode"])
    pcaFeatsFileName = getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='PCA')
    hogFeatsFileName = getFileName(dataToUse=dataToUse, normMode=normMode, pcaCount=-1, numOfSigns=numOfSigns, expectedFileType='Data')
    if not skipLoadOfOriginalData:
        feat_set, _, _ = loadData_hog(base_dir=base_dir, data_dir=data_dir, loadHogIfExist=True, numOfSigns=numOfSigns)
    else: #load the data
        feat_set = loadFileIfExist(data_dir, hogFeatsFileName)

    feat_set_pca = loadFileIfExist(data_dir, pcaFeatsFileName)
    if feat_set_pca.size == 0:
        feat_set_pca = applyPCA2Data(feat_set, data_dir, data_dim, dataToUse=dataToUse, numOfSigns=numOfSigns, loadIfExist=True)
    return feat_set_pca

def rnnGetDataByTimeSteps(X_in, frameIDs, timesteps):
    actualFrCnt, dimOfFeat = X_in.shape
    X_out = X_in[frameIDs, :].reshape(-1, timesteps, dimOfFeat)
    return X_out

def rnnPredictOverlappingDS(timesteps, detailed_labels_all, verbose=0):
    rnnPredictIDs = []
    rnnFrameIDsForLabelAcc = []
    fIDBase = 0
    validIDBase = 0
    uniqueSigns = np.unique(detailed_labels_all[:, 0])
    for s in uniqueSigns:
        signRows = np.argwhere(detailed_labels_all[:, 0] == s).squeeze()
        signLabs = detailed_labels_all[signRows, 1:]
        videoList = np.unique(signLabs[:, 0])
        for v in videoList:
            frameCnt = len(np.argwhere(signLabs[:, 0] == v).squeeze())

            vidIDsValid, frameIDsForLabelAcc = funcH.rnn_getValidIDs(frameCnt, timesteps, verbose=verbose)

            vidIDsValid += fIDBase
            frameIDsForLabelAcc += validIDBase

            rnnPredictIDs.append(vidIDsValid)
            rnnFrameIDsForLabelAcc.append(frameIDsForLabelAcc)

            fIDBase += frameCnt
            validIDBase += len(vidIDsValid)

    predictIDs = np.concatenate(rnnPredictIDs, axis=0)
    frameIDsForLabelAcc = np.concatenate(rnnFrameIDsForLabelAcc, axis=0)

    return predictIDs, frameIDsForLabelAcc

def getRNNTrainLabels_frameOverlap(timesteps, frameOverlap, detailed_labels_all, verbose=0):
    rnnTrainIDs = []
    uniqueSigns = np.unique(detailed_labels_all[:, 0])
    fIDBase = 0
    for s in uniqueSigns:
        signRows = np.argwhere(detailed_labels_all[:, 0] == s).squeeze()
        signLabs = detailed_labels_all[signRows, 1:]
        videoList = np.unique(signLabs[:, 0])
        for v in videoList:
            frameCnt = len(np.argwhere(signLabs[:, 0] == v).squeeze())

            vidIDsTrain, _ = funcH.rnn_getTrainIDs(frameCnt, timesteps, frameOverlap, verbose=verbose)
            vidIDsTrain += fIDBase
            rnnTrainIDs.append(vidIDsTrain)
            fIDBase += frameCnt

    rnnTrainIDs = np.concatenate(rnnTrainIDs, axis=0)

    return rnnTrainIDs

def getRNNTrainLabels_patchPerVideos(timesteps, patchFromEachVideo, detailed_labels_all, verbose=0):
    labels = []
    labels_detailed = []
    wCols = []
    fIDBase = 0
    uniqueSigns = np.unique(detailed_labels_all[:, 0])
    for s in uniqueSigns:
        signRows = np.argwhere(detailed_labels_all[:, 0] == s).squeeze()
        signLabs = detailed_labels_all[signRows, 1:]
        videoList = np.unique(signLabs[:, 0])
        for v in videoList:
            videoRows = np.argwhere(signLabs[:, 0] == v).squeeze()
            frameCnt = len(videoRows)

            toFrCnt = np.linspace(timesteps, frameCnt, num=patchFromEachVideo).astype(int)
            frFrCnt = toFrCnt-timesteps

            for p in range(patchFromEachVideo):
                fr = frFrCnt[p] + fIDBase
                to = toFrCnt[p] + fIDBase

                labs = np.array(detailed_labels_all[fr:to, 3])
                detailedLabels = np.array([s, v, p, fr, to])

                labels.append(labs)
                labels_detailed.append(detailedLabels)
                wCols.append(np.arange(fr, to))#actual ids e.g. 25023

            fIDBase += frameCnt

    labels = np.array(labels)
    labels_detailed = np.array(labels_detailed)
    trainIDs = np.concatenate(wCols, axis=0)
    return trainIDs, labels, labels_detailed

def getRNNTrainLabels_lookBack(look_back, dataSetLen):
    trainIDs = []
    blockCnt = int(dataSetLen/look_back)
    for i in range(blockCnt):
        fr = i*look_back
        to = (i+1)*look_back
        trainIDs.append(np.arange(fr, to))
    trainIDs = np.concatenate(trainIDs, axis=0)
    return trainIDs

def applyCorrespondance(feat_set_pca, corrFramesAll, corr_indis_a, applyCorr):
    if applyCorr >= 2:
        inFeats = feat_set_pca[corrFramesAll[corr_indis_a, :], :]
        outFeats = feat_set_pca[corrFramesAll[1 - corr_indis_a, :], :]
    else:
        inFeats = feat_set_pca
        outFeats = feat_set_pca
    return inFeats, outFeats

# dataset definition
class HandCraftedDataset(Dataset):
    # load the dataset
    def __init__(self, path, X=None, y=None):
        # load the csv file as a dataframe
        if X is None and y is None:
            df = pd.read_csv(path, header=None)
            # store the inputs and outputs
            self.X = df.values[:, :-1]
            self.y = df.values[:, -1]
            # ensure input data is floats
            self.X = self.X.astype('float32')
            # label encode target and ensure the values are floats
            self.y = LabelEncoder().fit_transform(self.y)
            #print(self.X)
            #print(self.y)
        else:
            self.X = X.astype('float32')
            self.y = y
        print(self.X.shape)
        print(self.y.shape)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])