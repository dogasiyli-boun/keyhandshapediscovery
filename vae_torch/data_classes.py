from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
from shutil import copyfile, rmtree
import numpy as np
import pandas as pd

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

class khs_dataset(Dataset):
    def __init__(self, root_dir, is_train, input_size, input_initial_resize=None, datasetname="hospdev"):
        self.root_dir = root_dir


        data_transform = \
        transforms.Compose([
            transforms.Resize(input_initial_resize),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]) if is_train and input_initial_resize is not None else \
        transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
        ])

        self.transform = data_transform
        self.datasetname = datasetname

        images, labels, ids = parseDataset_khs(root_dir)

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

def createDirIfNotExist(dir2create):
    if not os.path.isdir(dir2create):
        os.makedirs(dir2create)

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                f.endswith(endString) and
                os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

def getFolderList(dir2Search, startString="", endString="", sortList=False):
    folderList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                  f.endswith(endString) and
                  os.path.isdir(os.path.join(dir2Search, f))]
    if sortList:
        folderList = np.sort(folderList)
    return folderList

def create_sub_folders(targets, dir_path):
    """Creates empty folders which have the same name as given targets in dir_path"""
    for t in targets:
        createDirIfNotExist(os.path.join(dir_path, t))

def count_data_in_folder(data_path):
    if not os.path.isdir(data_path):
        print("data_path({:s}) doesnt exist".format(data_path))
        return [], []
    targets = getFolderList(dir2Search=data_path, sortList=True)
    img_cnt = np.zeros(np.shape(targets))
    for i, t in enumerate(targets):
        source_path = os.path.join(data_path, t)
        samples = getFileList(dir2Search=source_path, endString=".png")
        img_cnt[i] = len(samples)
    return targets, img_cnt

def create_data_folder(userIDTest, userIDValid, nos, to_folder, base_dir="/home/doga/DataFolder"):
    #  base_dir = funcH.getVariableByComputerName('base_dir')  # xx/DataPath or xx/DataFolder
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    data_path = os.path.join(base_dir, data_path_base, "imgs")  # original path of data to load
    data_ident = "te{:d}_va{:d}_nos{:d}".format(userIDTest, userIDValid, nos)
    train_path = os.path.join(to_folder, "conv_data_" + data_ident, 'data_tr')
    valid_path = os.path.join(to_folder, "conv_data_" + data_ident, 'data_va')
    test_path = os.path.join(to_folder, "conv_data_" + data_ident, 'data_te')

    createDirIfNotExist(train_path)
    createDirIfNotExist(valid_path)
    createDirIfNotExist(test_path)

    cnt_table_fileName = os.path.join(to_folder, "conv_data_" + data_ident,
                                      "cnt_table" + "_te{:d}_va{:d}_nos{:d}".format(userIDTest, userIDValid,
                                                                                    nos) + ".csv")
    targets = getFolderList(dir2Search=data_path, sortList=True).tolist()
    table_rows = targets.copy()
    table_rows.append("total")
    cnt_table = pd.DataFrame(index=table_rows, columns=["train", "validation", "test", "total"])
    for col in cnt_table.columns:
        cnt_table[col].values[:] = 0

    if os.path.isdir(train_path) and os.path.isdir(valid_path) and os.path.isdir(test_path):
        rmtree(train_path, ignore_errors=True)
        rmtree(valid_path, ignore_errors=True)
        rmtree(test_path, ignore_errors=True)

    create_sub_folders(targets, train_path)
    create_sub_folders(targets, valid_path)
    create_sub_folders(targets, test_path)
    for col in cnt_table.columns:
        cnt_table[col].values[:] = 0

    spaces_list = []
    for t in targets:
        print(f"Start copying target {t} -->")
        source_path = os.path.join(data_path, t)
        samples = getFileList(dir2Search=source_path, endString=".png")
        # according to user_id_dict
        cnt_table["total"][t] = len(samples)
        cnt_table["total"]["total"] += len(samples)
        train_samples = []
        for s in samples:
            sample_dict = s.split(sep="_")
            # <3 signID><1 userID><2 repID>
            # int_id = int(sample_dict[1])
            # user_id = ((int_id - int_id.__mod__(100))/100).__mod__(10)
            # user_id_str = sample_dict[1][3]
            user_id_int = int(sample_dict[1][3])
            # if user_id_dict["valid"] == user_id_int:
            #    copyfile(os.path.join(source_path, s), os.path.join(valid_path, t, s))
            #    cnt_table["validation"][t] += 1
            if userIDTest == user_id_int:
                copyfile(os.path.join(source_path, s), os.path.join(test_path, t, s))
                cnt_table["test"][t] += 1
            elif userIDValid == user_id_int:
                copyfile(os.path.join(source_path, s), os.path.join(valid_path, t, s))
                cnt_table["validation"][t] += 1
            else:
                copyfile(os.path.join(source_path, s), os.path.join(train_path, t, s))
                cnt_table["train"][t] += 1

        cnt_table["train"]["total"] += cnt_table["train"][t]
        cnt_table["validation"]["total"] += cnt_table["validation"][t]
        cnt_table["test"]["total"] += cnt_table["test"][t]
        print(
            f"Copied {t} --> train({cnt_table['train'][t]}),valid({cnt_table['validation'][t]}),test({cnt_table['test'][t]})")

    pd.DataFrame.to_csv(cnt_table, path_or_buf=cnt_table_fileName)
    print('\n'.join(map(str, spaces_list)))
    samples_list_filename = cnt_table_fileName.replace(".csv", "_sl.txt")
    with open(samples_list_filename, 'w') as f:
        for i, item in enumerate(spaces_list):
            f.write("%s - %s\n" % (str(targets[i]), str(item)))

    return data_ident