import os
import numpy as np
from skimage import data
from skimage.feature import hog
from sklearn.decomposition import PCA
import glob
import helperFuncs as funcH

def loadData_hog(base_dir, loadHogIfExist=True, hogFeatsFileName='hog_set.npy', labelsFileName='labels.npy', detailedLabelsFileName='detailed_labels.npy'):
    hogFeatsFileNameFull = base_dir + os.sep + hogFeatsFileName
    labelsFileNameFull = base_dir + os.sep + labelsFileName
    detailedLabelsFileNameFull = base_dir + os.sep + detailedLabelsFileName
    base_dir_train_feat = os.path.join(base_dir, 'neuralNetHandVideos')
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
        foldernames = os.listdir(base_dir_train_feat)
        signID = 0
        frameCount = 0
        for f in foldernames:
            sign_folder = os.path.join(base_dir_train_feat, str(f).format(':02d'))
            if not os.path.isdir(sign_folder):
                continue
            signID = signID + 1
            videoID = 0
            videos = os.listdir(sign_folder)
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
                hogFeats_curVideo_FileNameFull = video_folder + os.sep + hogFeatsFileName.replace('.npy', '.npz')
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
                        elif frame.endswith('.txt'):
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


def loopTroughFeatureSet(base_dir, data_dir, featType='sn'):
    base_dir_nn = os.path.join(base_dir, 'neuralNetHandVideos')

    detailedLabelsFileNameFull = data_dir + os.sep + 'detailed_labels.npy'
    detailedLabels_all = np.load(detailedLabelsFileNameFull)

    print(detailedLabels_all.shape)

    signNames = os.listdir(base_dir_nn)
    signID = 0
    corrFramesAll = np.array([])
    for signCur in signNames:
        corrFramesSign = np.array([])
        sign_folder = os.path.join(base_dir_nn, signCur)
        if not os.path.isdir(sign_folder):
            continue
        signID = signID + 1
        videos = os.listdir(sign_folder)
        vidCnt = len(videos)
        detailedLabels_all_sign_rows = np.argwhere(detailedLabels_all[:, 0] == signID).flatten()

        corrFramesSignFileName = sign_folder + os.sep + 'corrFrames_' + str(signID) + '.npy'
        if os.path.isfile(corrFramesSignFileName):
            corrFramesSign = np.load(corrFramesSignFileName)
            print('loaded corrFramesSign of shape ', corrFramesSign.shape)
        else:
            for v1 in range(0, vidCnt):
                print('s_', signCur, ', v(', videos[v1], ' to vCnt-', vidCnt-v1)
                for v2 in range(v1 + 1, vidCnt):
                    video_folder_1 = os.path.join(sign_folder, videos[v1])
                    video_folder_2 = os.path.join(sign_folder, videos[v2])
                    #frameList_v1 = os.path.join(sign_folder, videos[v1]) + os.sep + '*.png'
                    #frameList_v1 = glob.glob(frameList_v1)
                    #frameList_v2 = os.path.join(sign_folder, videos[v2]) + os.sep + '*.png'
                    #frameList_v2 = glob.glob(frameList_v2)
                    #frCnt_1 = len(np.sort(frameList_v1))
                    #frCnt_2 = len(np.sort(frameList_v2))

                    hog_v1 = video_folder_1 + os.sep + 'hog_set.npz'
                    npzfile = np.load(hog_v1)
                    feat_set_v1 = npzfile['feat_set_video']
                    #labels_v1 = npzfile['labels']
                    detailedLabels_cur_vid_rows_rel = np.argwhere(detailedLabels_all[detailedLabels_all_sign_rows, 1] == v1+1).flatten()
                    frIDs_v1 = detailedLabels_all_sign_rows[detailedLabels_cur_vid_rows_rel]

                    hog_v2 = video_folder_2 + os.sep + 'hog_set.npz'
                    npzfile = np.load(hog_v2)
                    feat_set_v2 = npzfile['feat_set_video']
                    #labels_v2 = npzfile['labels']
                    detailedLabels_cur_vid_rows_rel = np.argwhere(detailedLabels_all[detailedLabels_all_sign_rows, 1] == v2+1).flatten()
                    frIDs_v2 = detailedLabels_all_sign_rows[detailedLabels_cur_vid_rows_rel]

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
            print('saved corrFramesSign of shape ', corrFramesSign.shape)
        if corrFramesAll.size == 0:
            corrFramesAll = corrFramesSign
        else:
            corrFramesAll = np.hstack((corrFramesAll, corrFramesSign))
    return corrFramesAll, detailedLabels_all

def applyPCA2Data(feat_set, labels_all, base_dir, data_dim, loadIfExist = True, pcaFeatsFileName = 'feat_set_pca.npy', labelsFileName='labels.npy'):
    pcaFeatsFileNameFull = base_dir + os.sep + pcaFeatsFileName
    labelsFileNameFull = base_dir + os.sep + labelsFileName

    if labels_all.size==0:
        labels_all = np.load(labelsFileNameFull)
    if loadIfExist and os.path.isfile(pcaFeatsFileNameFull):
        print('loading feat_set_pca from(', pcaFeatsFileNameFull, ')')
        feat_set_pca = np.load(pcaFeatsFileNameFull)
        print('loaded feat_set_pca(',  feat_set_pca.shape, ') from(', pcaFeatsFileName, ')')
    else:
        #feat_set = loadData_hog(base_dir)
        pca = PCA(n_components=data_dim)
        feat_set_pca = pca.fit_transform(feat_set)
        np.save(pcaFeatsFileNameFull, feat_set_pca)
    return feat_set_pca, labels_all