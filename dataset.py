from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

def parseDataset_nnv(base_dir, labelsFileName='labels_convModel.npy', detailedLabelsFileName='detailed_labels_convModel.npy'):
    labelsFileNameFull = base_dir + os.sep + labelsFileName
    detailedLabelsFileNameFull = base_dir + os.sep + detailedLabelsFileName
    base_dir_train_feat = base_dir
    images = []
    labels = []
    ids = []
    detailedLabels_all = np.array([0, 0, 0, 0])
    labels_all = []
    foldernames = np.sort(os.listdir(base_dir_train_feat))
    signID = 0
    frameCount = 0
    im_id = 0
    for f in foldernames:
        sign_folder = os.path.join(base_dir_train_feat, str(f).format(':02d'))
        if not os.path.isdir(sign_folder):
            continue
        signID = signID + 1
        videoID = 0
        videos = np.sort(os.listdir(sign_folder))
        #print('going to grab images from sign folder(', sign_folder, ')')
        for v in videos:
            video_folder = os.path.join(sign_folder, v)
            if not os.path.isdir(video_folder):
                continue
            videoID = videoID + 1
            #print('going to grab images from video folder(', video_folder, ')')
            frames = os.listdir(video_folder)
            for frame in sorted(frames):
                if frame.endswith('.png'):
                    frame_name = os.path.join(video_folder, frame)
                    images.append(frame_name)
                    ids.append(im_id)
                    im_id = im_id + 1
                elif frame.endswith('_labels.txt'):
                    labels = np.loadtxt(os.path.join(video_folder, frame))

            fr = frameCount
            to = frameCount + len(labels)
            frCnt = to - fr
            frameIDs = np.asarray(range(fr, to)).reshape(frCnt, -1)
            detailedLabels_video = np.hstack((signID * np.ones([frCnt, 1]), videoID * np.ones([frCnt, 1]), frameIDs, np.asarray(labels).reshape(frCnt, -1)))

            for i in range(0, len(labels)):
                labels_all.append(int(labels[i]))

            if np.all(detailedLabels_all == 0):
                detailedLabels_all = detailedLabels_video
            else:
                detailedLabels_all = np.vstack((detailedLabels_all, detailedLabels_video))
            frameCount = len(labels_all)
    np.save(labelsFileNameFull, labels_all)
    np.save(detailedLabelsFileNameFull, detailedLabels_all)
    return images, labels_all, ids

def parseDataset_khs(root_dir, istrain):
    images = []
    labels = []
    ids = []

    bothFolder = root_dir + os.sep + 'both'
    singleFolder = root_dir + os.sep + 'single'
    bothKhsList = os.listdir(bothFolder)
    singleKhsList = os.listdir(singleFolder)

    valPerc = 0.1 #valiation percentage
    khsID_general = -1
    im_id = 0

    for khs_b_id in range(0,len(bothKhsList)):
        khsName = bothKhsList[khs_b_id]
        khsFolder = bothFolder + os.sep + khsName
        if os.path.isdir(khsFolder)==0:
            continue
        khsID_general = khsID_general + 1
        image_list = os.listdir(khsFolder)
        imCnt = len(image_list)
        val_after = np.floor(imCnt*(1-valPerc))

        for imID in range(0,imCnt):#range(0,len(class_list)):
            if imID > val_after and not istrain:
                images.append(khsFolder + os.sep + image_list[imID])
                labels.append(khsID_general)
                ids.append(im_id)
                im_id = im_id + 1
            elif istrain:
                images.append(khsFolder + os.sep + image_list[imID])
                labels.append(khsID_general)
                ids.append(im_id)
                im_id = im_id + 1


    for khs_s_id in range(0,len(singleKhsList)):
        khsName = singleKhsList[khs_s_id]
        khsFolder = singleFolder + os.sep + khsName
        if os.path.isdir(khsFolder)==0:
            continue
        khsID_general = khsID_general + 1
        image_list = os.listdir(khsFolder)
        imCnt = len(image_list)
        val_after = np.floor(imCnt*(1-valPerc))

        for imID in range(0,imCnt):#range(0,len(class_list)):
            if imID > val_after and not istrain:
                images.append(khsFolder + os.sep + image_list[imID])
                labels.append(khsID_general)
                ids.append(im_id)
                im_id = im_id + 1
            elif istrain:
                images.append(khsFolder + os.sep + image_list[imID])
                labels.append(khsID_general)
                ids.append(im_id)
                im_id = im_id + 1
    return images, labels, ids

def parseDataset_ds5(root_dir, istrain):
    images = []
    labels = []
    ids = []
    user_list = os.listdir(root_dir)
    for u in range(0, len(user_list)):
        class_list = os.listdir(root_dir + os.sep + user_list[u])
        for c in range(3):  # range(0,len(class_list)):
            image_list = os.listdir(os.path.join(root_dir, user_list[u], class_list[c]))
            for i in image_list:
                if u == 0 and not istrain:
                    images.append(root_dir + os.sep + user_list[u] + os.sep + class_list[c] + os.sep + i)
                    labels.append(c)
                    ids.append(im_id)
                    im_id = im_id + 1
                elif istrain:
                    images.append(root_dir + os.sep + user_list[u] + os.sep + class_list[c] + os.sep + i)
                    labels.append(c)
                    ids.append(im_id)
                    im_id = im_id + 1
    return images, labels, ids

def parseDataset(root_dir, datasetname, istrain):
    if datasetname == 'ds5':
        images, labels, ids = parseDataset_ds5(root_dir, istrain)
    elif datasetname == 'khs':
        images, labels, ids = parseDataset_khs(root_dir, istrain)
    elif datasetname == 'nnv':
        #root_dir = '/home/doga/DataFolder/neuralNetHandVideos'#str.replace(root_dir,'neuralNetHandImages','neuralNetHandVideos')
        images, labels, ids = parseDataset_nnv(root_dir)
    return images, labels, ids

class HandShapeDataset(Dataset):
    def __init__(self, root_dir, istrain=True, transform=None, datasetname='ds5'):
        self.root_dir = root_dir
        self.transform = transform
        self.datasetname = datasetname

        images, labels, ids = parseDataset(root_dir, datasetname, istrain)

        self.images = images
        self.labels = labels
        self.ids = ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        ids = self.ids[idx]
        sample = {'image': image, 'label': label, 'id': ids}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def updateLabels(self, labels):
        self.labels = list(np.array(labels).astype('int'))

    def _len_(self):
        return len(self.labels)

if __name__ == '__main__':
    x = 5 #do nothing basically
    #inRootDir = '/home/doga/Desktop'+os.sep+'ds5'
    #f = HandShapeDataset(root_dir=inRootDir, istrain=True, datasetname)

