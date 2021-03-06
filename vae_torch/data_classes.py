from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
from shutil import copyfile, rmtree
import numpy as np
import pandas as pd
from helperFuncs import get_mapped_0_k_indices, createDirIfNotExist, getFileList, getFolderList
import projRelatedHelperFuncs as prHF
from torch import from_numpy as torch_from_numpy

#fashion mnist
from torchvision.datasets import MNIST as MNISTds
from torchvision.datasets import FashionMNIST as fashionMNISTds
from torchvision.datasets import CIFAR10 as CIFAR10ds
from torchvision.utils import save_image

def get_def_from_im_name_hospisign(frame_name, verbose=0):
    khs_name, id_str, hand_char, fr_id_str_rel, fr_id_abs = frame_name.split('_')
    signID = int(id_str[0:3])
    signerID = int(id_str[3])
    repID = int(id_str[4:6])
    hand_int = 0 if hand_char == 'L' else 1 if hand_char == 'R' else 2
    fr_id_str_rel = int(fr_id_str_rel)
    fr_id_str_vid_khs = int(fr_id_abs.replace('.png', ''))
    if verbose > 0:
        print("khs_name=", khs_name)
        print("signID=",signID)
        print("signerID=",signerID)
        print("repID=", repID)
        print("hand_char=", hand_char)
        print("hand_int=", hand_int)
        print("fr_id_str_rel=", fr_id_str_rel)
        print("fr_id_str_vid_khs=", fr_id_str_vid_khs)

    frame_info = {
        "khs_name": khs_name,
        "signID": signID,
        "signerID": signerID,
        "repID": repID,
        "hand_char": hand_char,
        "hand_int": hand_int,
        "fr_id_str_rel": fr_id_str_rel,
        "fr_id_str_vid_khs": fr_id_str_vid_khs
    }
    return frame_info

def parseDataset_khs_v2(root_dir):
    images = []
    labels = []
    ids = []
    khs_groups = []
    khs_names = []
    sign_ids = []
    signer_ids = []
    hand_ids = []

    khs_list = os.listdir(root_dir)
    khs_unique_idx = np.zeros(len(khs_list),dtype=np.uint8)
    im_id = 0

    for khs_id in range(0, len(khs_list)):
        khs_name = khs_list[khs_id]
        khs_folder = os.path.join(root_dir, khs_name)
        if os.path.isdir(khs_folder) == 0:
            continue
        image_list = os.listdir(khs_folder)
        im_cnt = len(image_list)

        khs_unique_idx[khs_id] = im_id

        for imID in range(0, im_cnt):
            images.append(os.path.join(khs_folder, image_list[imID]))
            labels.append(khs_id)
            ids.append(im_id)
            im_id = im_id + 1
            frame_info = get_def_from_im_name_hospisign(image_list[imID], verbose=0)
            khs_groups.append(khs_name)
            khs_names.append(frame_info["khs_name"])
            sign_ids.append(frame_info["signID"])
            signer_ids.append(frame_info["signerID"])
            hand_ids.append(frame_info["hand_int"])

    sign_ids_map = get_mapped_0_k_indices(sign_ids, verbose=0)
    signer_ids_map = get_mapped_0_k_indices(signer_ids, verbose=0)
    hand_ids_map = get_mapped_0_k_indices(hand_ids, verbose=0)

    khs_ids_map = {
        "unique": np.asarray(khs_list),
        "unique_idx": khs_unique_idx,
        "mapped": labels
    }
    base_dict = {
        "image_paths": images,
        "labels": labels,
        "ids": ids,
        "khs_groups": khs_groups,
        "khs_names": khs_names,
        "sign_ids": sign_ids,
        "signer_ids": signer_ids,
        "hand_ids": hand_ids,
    }
    extra_dict = {
        "khs_ids_map": khs_ids_map,
        "sign_ids_map": sign_ids_map,
        "signer_ids_map": signer_ids_map,
        "hand_ids_map": hand_ids_map,
    }

    return base_dict, extra_dict

class khs_dataset_v2(Dataset):
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

        base_dict, extra_dict = parseDataset_khs_v2(root_dir)

        self.image_paths = base_dict["image_paths"]
        self.labels = base_dict["labels"]
        self.ids = base_dict["ids"]
        self.khs_groups = base_dict["khs_groups"]
        self.khs_names = base_dict["khs_names"]
        self.sign_ids = base_dict["sign_ids"]
        self.signer_ids = base_dict["signer_ids"]
        self.hand_ids = base_dict["hand_ids"]
        self.labels_extra = extra_dict

        self.available_label_types = ["khs", "sign", "signer", "hand"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        ids = self.ids[idx]
        khs_group_str = self.khs_groups[idx]
        khs_name_str = self.khs_names[idx]
        sign_id = self.sign_ids[idx]
        signer_ids = self.signer_ids[idx]
        hand_ids = self.hand_ids[idx]
        label_extra = {
            "khs": self.labels_extra["khs_ids_map"]["mapped"][idx],
            "sign": self.labels_extra["sign_ids_map"]["mapped"][idx],
            "signer": self.labels_extra["signer_ids_map"]["mapped"][idx],
            "hand": self.labels_extra["hand_ids_map"]["mapped"][idx],
        }
        sample = {
            'image': image, 'label': label,  'id': ids,
            'khs_group_str': khs_group_str, 'khs_name_str': khs_name_str,
            'sign_id': sign_id, "signer_id": signer_ids,  "hand_id": hand_ids,
            "label_extra": label_extra
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def _len_(self):
        return len(self.labels)

class khs_dataset_hand_crafted(Dataset):
    def __init__(self, datasetname):
        dataIdent, pca_dim, nos = str(datasetname).split('_')
        x, labels_all, labels_sui, labels_map = prHF.combine_pca_hospisign_data(dataIdent=dataIdent,
                                                                                pca_dim=int(pca_dim),
                                                                                nos=int(nos), verbose=2)
        self.data = x
        self.labels = np.asarray(labels_all, dtype=int).squeeze()
        self.ids = np.asarray(np.arange(0, len(self.labels)), dtype=int)
        self.label_names = np.asarray(labels_map["khsName"])
        self.datasetname = datasetname

        # self.sign_ids = base_dict["sign_ids"]
        # self.signer_ids = base_dict["signer_ids"]
        # self.hand_ids = base_dict["hand_ids"]

        self.available_label_types = ["khs"]  # , "sign", "signer", "hand"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vector = torch_from_numpy(self.data[idx, :]).float()
        label = self.labels[idx]
        ids = self.ids[idx]
        # khs_group_str = self.khs_groups[idx]
        # khs_name_str = self.khs_names[idx]
        # sign_id = self.sign_ids[idx]
        # signer_ids = self.signer_ids[idx]
        # hand_ids = self.hand_ids[idx]
        #label_extra = {
        #    "khs": self.labels_extra["khs_ids_map"]["mapped"][idx],
        #    "sign": self.labels_extra["sign_ids_map"]["mapped"][idx],
        #    "signer": self.labels_extra["signer_ids_map"]["mapped"][idx],
        #    "hand": self.labels_extra["hand_ids_map"]["mapped"][idx],
        #}
        sample = {
            'vector': vector, 'label': label,  'id': ids,
        #    'khs_group_str': khs_group_str, 'khs_name_str': khs_name_str,
        #    'sign_id': sign_id, "signer_id": signer_ids,  "hand_id": hand_ids,
        #    "label_extra": label_extra
        }
        return sample

    def _len_(self):
        return len(self.labels)

def data_transform(input_size, input_initial_resize, is_train, load_train_as_test, flatten, transpose_final):
    if flatten:
        return transpose_final
    #  self.transform = transforms.Compose([transforms.ToTensor()])
    _transform = transforms.Compose([transforms.Resize(input_size), transpose_final])
    if input_initial_resize is not None and is_train:
        _transform = transforms.Compose([
            transforms.Resize(input_initial_resize),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transpose_final
        ])
    elif input_initial_resize is None and is_train and not load_train_as_test:
        _transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transpose_final
        ])
    return _transform
class fashion_mnist(Dataset):
    def __init__(self, fashionMNISTds_fold, is_train, input_size, flatten=False, input_initial_resize=None, load_train_as_test=False, datasetname="fashion_mnist"):
        self.root_dir = fashionMNISTds_fold
        self.flatten = flatten
        transpose_final = transforms.Compose([transforms.ToTensor()])
        self.transform = data_transform(input_size, input_initial_resize, is_train, load_train_as_test, flatten, transpose_final)
        self.datasetname = datasetname
        dataset = fashionMNISTds(
            root=os.path.join(fashionMNISTds_fold),
            train=is_train,
            download=True
        )
        img_all = []
        lab_all = []
        for i in range(0, len(dataset)):
            img, lb = dataset[i]
            img_all.append(img)
            lab_all.append(lb)

        self.images = img_all
        self.labels = lab_all

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        ids = idx
        sample = {'image': image, 'label': label, 'ids': ids}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.flatten:
            sample['image'] = sample['image'].view(sample['image'].size(0), -1)
        return sample

    def _len_(self):
        return len(self.labels)

    @staticmethod
    def save_decoded_image(img, name, input_size=28):
        img = img.view(img.size(0), 1, input_size, input_size)
        save_image(img, name)

class cifar10(Dataset):
    def __init__(self, cifar10ds_fold, is_train, input_size, flatten=False, input_initial_resize=None, load_train_as_test=False, datasetname="cifar10"):
        self.root_dir = cifar10ds_fold
        self.input_size = input_size
        self.flatten = flatten
        transpose_final = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = data_transform(input_size, input_initial_resize, is_train, load_train_as_test, flatten, transpose_final)
        self.datasetname = datasetname
        dataset = CIFAR10ds(
            root=os.path.join(cifar10ds_fold),
            train=is_train,
            download=True
        )
        img_all = []
        lab_all = []
        for i in range(0, len(dataset)):
            img, lb = dataset[i]
            img_all.append(img)
            lab_all.append(lb)

        self.images = img_all
        self.labels = lab_all

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        ids = idx
        sample = {'image': image, 'label': label, 'ids': ids}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def _len_(self):
        return len(self.labels)

    @staticmethod
    def save_decoded_image(img, name, input_size=32):
        img = img.view(img.size(0), 1, input_size, input_size)
        save_image(img, name)

class mnist(Dataset):
    def __init__(self, MNISTds_fold, is_train, input_size, flatten=False, input_initial_resize=None, load_train_as_test=False, datasetname="mnist"):
        self.root_dir = MNISTds_fold
        self.flatten = flatten
        transpose_final = [transforms.ToTensor()]
        self.transform = data_transform(input_size, input_initial_resize, is_train, load_train_as_test, flatten, transpose_final)
        self.datasetname = datasetname
        dataset = MNISTds(
            root=os.path.join(MNISTds_fold),
            train=is_train,
            download=True
        )
        img_all = []
        lab_all = []
        for i in range(0, len(dataset)):
            img, lb = dataset[i]
            img_all.append(img)
            lab_all.append(lb)

        self.images = img_all
        self.labels = lab_all

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        ids = idx
        sample = {'image': image, 'label': label, 'ids': ids}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.flatten:
            sample['image'] = sample['image'].view(sample['image'].size(0), -1)
        return sample

    def _len_(self):
        return len(self.labels)

    @staticmethod
    def save_decoded_image(img, name, input_size=28):
        img = img.view(img.size(0), 1, input_size, input_size)
        save_image(img, name)

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