import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
from skimage import data

def loadData_nnVidImages(dataDir, labelsFileName='vidLabels.npy', detailedLabelsFileName='detailed_vid_labels.npy'):
    labelsFileNameFull = dataDir + os.sep + labelsFileName
    detailedLabelsFileNameFull = dataDir + os.sep + detailedLabelsFileName
    base_dir_train_feat = os.path.join(dataDir, 'neuralNetHandVideos')

    detailedLabels_all = np.array([0, 0, 0, 0])
    labels_all = np.array([0, 0, 0, 0])
    feat_set = np.array([0, 0, 0, 0])
    foldernames = os.listdir(base_dir_train_feat)
    signID = 0
    frameCount = 0
    for f in sorted(foldernames):
        sign_folder = os.path.join(base_dir_train_feat, str(f).format(':02d'))
        if not os.path.isdir(sign_folder):
            continue
        signID = signID + 1
        videoID = 0
        videos = os.listdir(sign_folder)
        print(f)
        print('going to create hog from sign folder(', sign_folder, ')')
        for v in sorted(videos):
            video_folder = os.path.join(sign_folder, v)
            if not os.path.isdir(video_folder):
                continue
            videoID = videoID + 1
            print('going to create hog from video folder(', video_folder, ')')
            frames = os.listdir(video_folder)
            feat_set_video = np.array([0, 0, 0, 0])
            for frame in sorted(frames):
                if frame.endswith('.png'):
                    frame_name = os.path.join(video_folder, frame)
                    img = data.load(frame_name)
                    feat_current = np.expand_dims(np.expand_dims(img,axis=2),axis=0)
                    if np.all(feat_set_video == 0):
                        feat_set_video = feat_current
                    else:
                        feat_set_video = np.append(feat_set_video, feat_current, axis=0)
                elif frame.endswith('.txt'):
                    labels = np.loadtxt(os.path.join(video_folder, frame))
            fr = frameCount
            to = frameCount + len(labels)
            frCnt = to - fr
            frameIDs = np.asarray(range(fr, to)).reshape(frCnt, -1)
            detailedLabels_video = np.hstack((signID * np.ones([frCnt, 1]), videoID * np.ones([frCnt, 1]), frameIDs, np.asarray(labels).reshape(frCnt, -1)))

            if np.all(feat_set == 0):
                feat_set = feat_set_video
            else:
                feat_set = np.append(feat_set, feat_set_video, axis=0)

            if np.all(labels_all == 0):
                labels_all = labels
            else:
                labels_all = np.hstack((labels_all, labels))

            if np.all(detailedLabels_all == 0):
                detailedLabels_all = detailedLabels_video
            else:
                detailedLabels_all = np.vstack((detailedLabels_all, detailedLabels_video))
            frameCount = len(labels_all)
    np.save(labelsFileNameFull, labels_all)
    np.save(detailedLabelsFileNameFull, detailedLabels_all)
    return feat_set, labels_all, detailedLabels_all

def train_images(imDirectoryFolders):
    gen = ImageDataGenerator()
    train_im = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=False)
    train_generator = train_im.flow_from_directory(
            imDirectoryFolders,
             target_size=(224, 224),
             color_mode='rgb',
             batch_size=100,
             shuffle = True,
             class_mode='categorical')
    x = train_generator
    return x[0][0], x[0][1]

def loadMNISTData():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))  # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels