from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

def parseDataset_khs(root_dir):
    images = []
    labels = []
    ids = []

    khsList = os.listdir(root_dir)
    im_id = 0

    for khs_id in range(0,len(khsList)):
        khsName = khsList[khs_id]
        khsFolder = os.path.join(root_dir, khsName)
        if os.path.isdir(khsFolder)==0:
            continue
        image_list = os.listdir(khsFolder)
        imCnt = len(image_list)

        for imID in range(0, imCnt):#range(0,len(class_list)):
            images.append(os.path.join(khsFolder, image_list[imID]))
            labels.append(khs_id)
            ids.append(im_id)
            im_id = im_id + 1

    return images, labels, ids

def parseDataset(root_dir, datasetname):
    if datasetname == 'user_independent_khs':
        images, labels, ids = parseDataset_khs(root_dir)
    return images, labels, ids

class khs_dataset(Dataset):
    def __init__(self, root_dir, transform=None, datasetname='ds5'):
        self.root_dir = root_dir
        self.transform = transform
        self.datasetname = datasetname

        images, labels, ids = parseDataset(root_dir, datasetname)

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

    def _len_(self):
        return len(self.labels)

if __name__ == '__main__':
    x = 5 #do nothing basically
    #inRootDir = '/home/doga/Desktop'+os.sep+'ds5'
    #f = HandShapeDataset(root_dir=inRootDir, istrain=True, datasetname)

