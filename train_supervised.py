import datetime
import getopt
import os
import socket
import sys
import time

import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

import helperFuncs as funcH
from khs_dataset import khs_dataset

from shutil import copyfile, rmtree
import pandas as pd

# Freeze layers
def freeze_layers(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_device():
    # use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    return device

def extract_features(layerSize, model, feature_layer_string, images):
    model.eval()
    feature_layer = model._modules.get(feature_layer_string)  # for example avgpool
    features = []
    for input in images:
        model_input = input.view(-1, 3, 224, 224)
        model_input = model_input.to(get_device())
        feature = torch.zeros(1, layerSize)
        with torch.set_grad_enabled(False):
            # define a hook function
            def copy_data(m, i, o):
                feature.copy_(o.flatten(1).data)
            h = feature_layer.register_forward_hook(copy_data)
            model(model_input)
            h.remove()
        features.append(feature.cpu().numpy()[0])
    return features

def runTrainDs(model, optimizer, dsLoad_train):
    print("running --> runTrainDs", datetime.datetime.now().strftime("%H:%M:%S"))
    t = time.time()
    running_acc = []
    idShuffle = []
    cnt = 0
    for sample in dsLoad_train:
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        ids = sample['id'].cuda()
        cnt += len(labels)

        optimizer.zero_grad()
        outputs = model(images)
        crit = torch.nn.CrossEntropyLoss()
        loss = crit(outputs, labels)
        loss.backward()
        optimizer.step()

        _, indices = outputs.max(1)
        #try:
        acc = ((indices == labels).cpu().numpy().astype(dtype=np.float))
        for x in acc:
            running_acc.append(x)
        #except:
        #    print("cnt:", cnt, ", labels:", labels, ", indices:", indices)
        #    print("Unexpected error:", sys.exc_info()[0])
        #    raise
        #    os.exit(1)
        idShuffle = idShuffle + ids.tolist()

    tr_acc_run = (np.sum(np.array(running_acc)) / len(np.array(running_acc))).astype(dtype=np.float)
    elapsed = time.time() - t

    funcH.removeLastLine()
    print('runTrainDs completed (', funcH.getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))

    return tr_acc_run, idShuffle

def runValidDs(model, dsLoad_valid_test, return_feats=True, layerSize=512, dataIdentStr=""):
    epoc_acc = []
    features_avgPool = []
    predictions = []
    labels_all = []

    print("running --> runValidDs(", dataIdentStr, "return_feats=", str(return_feats), ", layerSize=", str(layerSize), ")", datetime.datetime.now().strftime("%H:%M:%S"))
    t = time.time()

    for sample in dsLoad_valid_test:
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        # ids = sample['id'].cuda()
        outputs = model(images)

        x, indices = outputs.max(1)
        acc = ((indices == labels).cpu().numpy().astype(dtype=np.float))
        epoc_acc += acc.tolist()
        #for x in acc:
        #  epoc_acc.append(x)

        if return_feats:
            feats = extract_features(layerSize=layerSize, model=model, feature_layer_string='avgpool', images=images)
            features_avgPool = features_avgPool + feats
        predictions += indices.tolist()
        labels_all += labels.tolist()

    elapsed = time.time() - t
    acc = (np.sum(np.array(epoc_acc)) / len(np.array(epoc_acc))).astype(dtype=np.float)

    funcH.removeLastLine()
    print('runValidDs(return_feats=', str(return_feats), ' completed (', funcH.getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))

    return acc, predictions, labels_all, features_avgPool

def parse_args_helper_01(paramsAll, argv):
    argSetDescriptions = ""
    go_00 = "hi:o:"
    go_01 = ["help="]

    excepStr = "train_supervised.py "
    for param in paramsAll:
        excepStr += "-" + param["vs"] + "<--" + param["paramName"] + "> "
        go_00 += param["vs"].replace("-", "") + ":"
        go_01.append(param["paramName"] + "=")
        argSetDescriptions += param["paramName"] + "=" + param["possibleValues"] + ","
    try:
      opts, args = getopt.getopt(argv[1:], go_00, go_01)
    except getopt.GetoptError:
        print(excepStr)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(excepStr)
            sys.exit(0)
        else:
            for param in paramsAll:
                if opt in (param["vs"], "--"+param["paramName"]):
                    if param["paramType"] == "int":
                        param["defaultVal"] = int(arg)
                    elif param["paramType"] == "str":
                        param["defaultVal"] = str(arg)
                    elif param["paramType"] == "bool":
                        if funcH.is_number(arg):
                            param["defaultVal"] = bool(int(arg))
                        else:
                            param["defaultVal"] = bool(arg)
                    elif param["paramType"] == "float":
                        param["defaultVal"] = float(arg)
                    else:
                        sys.exit(5)
                    param["dvSet"] = False

def parse_args_helper_02(paramsAll):
    valuesParamsCur = {}
    dvSetParamsCur = {}
    for i in range(len(paramsAll)):
        valuesParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["defaultVal"]
        dvSetParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["dvSet"]

    def setByParamName(paramsAll,curParamName,valuesParamsCur,dvSetParamsCur):
        for i in range(len(paramsAll)):
            if paramsAll[i]["paramName"] == curParamName:
                paramsAll[i]["defaultVal"] = valuesParamsCur[curParamName]
                paramsAll[i]["dvSet"] = dvSetParamsCur[paramsAll[i]["paramName"]]
                return paramsAll

    for param in paramsAll:
        paramsAll = setByParamName(paramsAll, param["paramName"], valuesParamsCur, dvSetParamsCur)

    def addValStr(defaultValsStr, userSetValsStr, param, endOfStr):
        if param["dvSet"]:
            defaultValsStr += param["paramName"] + "(" + str(param["defaultVal"]) + ")" + endOfStr
        else:
            userSetValsStr += param["paramName"] + "(" + str(param["defaultVal"]) + ")" + endOfStr
        return defaultValsStr, userSetValsStr

    defaultValsStr = " "
    userSetValsStr = " "
    for param in paramsAll:
        defaultValsStr, userSetValsStr = addValStr(defaultValsStr, userSetValsStr, param, ", ")
    print("Default values are set : ", os.linesep, defaultValsStr)
    print("Values set by user : ", os.linesep, userSetValsStr)

    for i in range(len(paramsAll)):
        valuesParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["defaultVal"]
    return valuesParamsCur, dvSetParamsCur

#updated
def parseArgs(argv):
    # train_supervised.py --modelName resnet18 --epochs 50
    #                --batch_size 16 --appendEpochBinary 1 --randomSeed 1
    #                --data_path_base neuralNetHandImages_nos1_rs224
    #                --userIDTest 4 --userIDValid 3
    param01 = {"paramName": "modelName", "possibleValues": "{'resnet18','vgg16','rsa','corsa'}",
               "vs": "-mn", "defaultVal": "resnet18", "dvSet": True, "paramType": "str"}
    param02 = {"paramName": "use_updated_model", "possibleValues": "{True or False",
               "vs": "-ua", "defaultVal": 1, "dvSet": True, "paramType": "bool"}
    param03 = {"paramName": "epochs", "possibleValues": "{50,200,500}",
               "vs": "-ep", "defaultVal": 50, "dvSet": True, "paramType": "int"}
    param04 = {"paramName": "batch_size", "possibleValues": "{0-B}",
               "vs": "-bs", "defaultVal": 16, "dvSet": True, "paramType": "int"}
    param05 = {"paramName": "appendEpochBinary", "possibleValues": "{0, 1}",
               "vs": "-ea", "defaultVal": 0, "dvSet": True, "paramType": "int"}
    param06 = {"paramName": "randomSeed", "possibleValues": "{some integer}",
               "vs": "-rs", "defaultVal": 1, "dvSet": True, "paramType": "int"}
    param07 = {"paramName": "data_path_base", "possibleValues": "{'any folder in DataFolder'}",
               "vs": "-dp", "defaultVal": None, "dvSet": True, "paramType": "str"}
    param08 = {"paramName": "userIDTest", "possibleValues": "{2-3-4-5-6-7}",
               "vs": "-ut", "defaultVal": 4, "dvSet": True, "paramType": "int"}
    param09 = {"paramName": "crossValidID", "possibleValues": "{1-2-3-4-5}",
               "vs": "-cv", "defaultVal": 1, "dvSet": True, "paramType": "int"}

    # model and data parameters
    paramsAll = [param01, param02, param03, param04, param05,
                 param06, param07, param08, param09]

    parse_args_helper_01(paramsAll, argv)
    valuesParamsCur, dvSetParamsCur = parse_args_helper_02(paramsAll)

    exp_ident = "_te{:d}_cv{:d}_".format(valuesParamsCur["userIDTest"], valuesParamsCur["crossValidID"]) + \
                valuesParamsCur["modelName"] + \
                valuesParamsCur["data_path_base"]

    params_dict = {
        "modelName": valuesParamsCur["modelName"],
        "use_updated_model": valuesParamsCur["use_updated_model"],
        "epochs": valuesParamsCur["epochs"],
        "batch_size": valuesParamsCur["batch_size"],
        "appendEpochBinary": valuesParamsCur["appendEpochBinary"],
        "randomSeed": valuesParamsCur["randomSeed"],
        "data_path_base": valuesParamsCur["data_path_base"],
        "exp_ident": exp_ident,
        "hostName": socket.gethostname(),
    }
    user_id_dict = {
        "cross_valid_id": valuesParamsCur["crossValidID"],
        "test": valuesParamsCur["userIDTest"],
    }

    return params_dict, user_id_dict

def initSomeVals(params_dict):
    useModelName = params_dict["modelName"]
    batch_size = params_dict["batch_size"]
    if useModelName == 'resnet18' or useModelName == 'resnet34' or useModelName == 'resnet50' or useModelName == 'resnet101':
        input_initial_resize = 256
        input_size = 224
        batch_size = 8
    if useModelName == 'resnet50' or useModelName == 'resnet101':
        input_initial_resize = 256
        input_size = 224
        batch_size = 10
    if useModelName == 'vgg16':
        input_initial_resize = 256
        input_size = 224

    num_workers = 4
    if 'dgx-server' in params_dict["hostName"]:
        num_workers = 4

    return input_initial_resize, input_size, batch_size, num_workers

def iterate_1(model, ds_loader, num_ftrs, ep, epochTo, epochStartTime, path_dict):
    model.eval()  # Set model to evaluation mode
    acc_tra, pred_tra, labels_tra, _ = runValidDs(model, ds_loader["train_te"], return_feats=False, layerSize=num_ftrs, dataIdentStr="train")
    acc_val, pred_val, labels_val, _ = runValidDs(model, ds_loader["valid"], return_feats=False, layerSize=num_ftrs, dataIdentStr="validation")
    acc_tes, pred_tes, labels_tes, _ = runValidDs(model, ds_loader["test"],  return_feats=False, layerSize=num_ftrs, dataIdentStr="test")

    result_row = np.array([ep, acc_tra, acc_val, acc_tes])
    print('ep={:d}/{:d}, acc_tra={:0.5f}, acc_val={:0.2f}, acc_tes={:0.2f}'.format(ep, epochTo, acc_tra, acc_val, acc_tes))
    print('Epoch done in (', funcH.getElapsedTimeFormatted(time.time() - epochStartTime), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))
    print('*-*-*-*-*-*-*')

    results_dict = {
        "labels_tra": labels_tra,
        "labels_val": labels_val,
        "labels_tes": labels_tes,
        "pred_tra": pred_tra,
        "pred_val": pred_val,
        "pred_tes": pred_tes,
    }
    resultFileNameFull = os.path.join(path_dict["result_fold"], "ep{:03d}.npy".format(ep))
    np.save(resultFileNameFull, results_dict, allow_pickle=True)

    return result_row

def saveToResultMatFile(result_csv, result_row):
    f = open(result_csv, 'a')
    np.savetxt(f, np.array(result_row).reshape(1, -1), fmt='%4.3f', delimiter='*', newline=os.linesep, header='', footer='', comments='', encoding=None)
    f.close()
    return

def getTransformFuncs(input_size, input_initial_resize):
    train_data_transform = transforms.Compose([
        transforms.Resize(input_initial_resize),
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_data_transform = transforms.Compose([
        transforms.Resize(input_size),
        #transforms.RandomSizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_data_transform, valid_data_transform

#needs update
def setEpochBounds(result_csv_file, num_epochs, appendEpochBinary):
    if not os.path.isfile(result_csv_file):
        completedEpochCnt = 0
    else:
        result_pd = pd.read_csv(result_csv_file, header=0, sep="*", names=["epoch", "train", "validation", "test"])
        completedEpochCnt = int(result_pd["epoch"].values[-1]+1)
    epochFr = completedEpochCnt
    epochTo = num_epochs + appendEpochBinary*completedEpochCnt
    if epochFr >= epochTo:
        print('epochFr(', str(epochFr), ') >= (', str(epochTo), ')epochTo. hence exiting.')
        sys.exit(5)
    print('epochFr = ', str(epochFr), ', epochTo = ', str(epochTo))
    return epochFr, epochTo

def getModel(params_dict, path_dict, expName, num_classes):

    useModelName = params_dict["modelName"]  # 'resnet18'
    useUpdatedModel = params_dict["use_updated_model"]  # True

    downloadedModelFile = os.path.join(path_dict["models"], 'model_' + useModelName + '_Down.model')
    updatedModelFile = os.path.join(path_dict["models"], 'model' + expName + '.model')

    if os.path.isfile(updatedModelFile) and useUpdatedModel:
        print('model(', useModelName, ') is being loaded from updatedModelFile')
        model = torch.load(f=updatedModelFile)
        print('model(', useModelName, ') has been loaded from updatedModelFile')
        print('final layer has (', str(model.fc.in_features), ') classes ?= ', str(num_classes))
    elif os.path.isfile(downloadedModelFile):
        print('model(', useModelName, ') is being loaded from downloadedModelFile')
        model = torch.load(f=downloadedModelFile)
        print('model(', useModelName, ') has been loaded from downloadedModelFile')
        print('final layer will be changed from (', str(model.fc.in_features), ') to ', str(num_classes))
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        print('final layer has (', str(model.fc.in_features), ') classes ?= ', str(num_classes))
    else:
        if useModelName == 'resnet18' or useModelName == 'resnet34' or useModelName == 'resnet50' or useModelName == 'resnet101':
            if useModelName == 'resnet18':
                model = models.resnet18(pretrained=True)
            elif useModelName == 'resnet34':
                model = models.resnet34(pretrained=True)
            elif useModelName == 'resnet50':
                model = models.resnet50(pretrained=True)
            elif useModelName == 'resnet101':
                model = models.resnet101(pretrained=True)
            print('model(', useModelName, ') has been downloaded and being saved to ', downloadedModelFile)
            torch.save(model, f=downloadedModelFile)
            print('final layer will be changed from (', str(model.fc.in_features), ') to ', str(num_classes))
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif useModelName == 'vgg16':
            model = models.vgg16(pretrained=True)
            print('model(', useModelName, ') has been downloaded and being saved to ', downloadedModelFile)
            torch.save(model, f=downloadedModelFile)
            print('final layer will be changed from (', str(model.fc.in_features), ') to ', str(num_classes))
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    freeze_layers(model, False)  # To freeze training weights or not

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer, updatedModelFile

def saveFeatsExtracted(data_dir, epochID, modelName, expName, featVec, labels, predictions):
    if epochID == 0:
        saveToFileName = os.path.join(data_dir, modelName + '_featsVec.npz')
    else:
        saveToFileName = os.path.join(data_dir, expName + '_featsVec.npz')

    actionStr = 'Updat' if os.path.isfile(saveToFileName) else 'Sav'
    print(actionStr + 'ing ', saveToFileName, ', feats', np.array(featVec).shape, ', labels and preds', np.array(labels).shape)
    np.savez(saveToFileName, feats=featVec, labels=labels, predictions=predictions)
    print(actionStr + 'ed ', saveToFileName)
    return

def create_sub_folders(targets, dir_path):
    """Creates empty folders which have the same name as given targets in dir_path"""
    for t in targets:
        funcH.createDirIfNotExist(os.path.join(dir_path, t))

def count_data_in_folder(data_path):
    if not os.path.isdir(data_path):
        print("data_path({:s}) doesnt exist".format(data_path))
        return [], []
    targets = funcH.getFolderList(dir2Search=data_path, sortList=True)
    img_cnt = np.zeros(np.shape(targets))
    for i, t in enumerate(targets):
        source_path = os.path.join(data_path, t)
        samples = funcH.getFileList(dir2Search=source_path, endString=".png")
        img_cnt[i] = len(samples)
    return targets, img_cnt

def validate_downloaded_data(data_path):
    if not os.path.isdir(data_path):
        print("data_path({:s}) doesnt exist".format(data_path))
        return False, [], []
    img_counts_matches = True
    targets = funcH.getFolderList(dir2Search=data_path, sortList=True).tolist()
    csv_file = funcH.getFileList(dir2Search=data_path, startString="cnt_table", endString=".csv")
    csv_file_exist = csv_file != []
    if csv_file_exist:
        cnt_pd = pd.read_csv(filepath_or_buffer=os.path.join(data_path, csv_file[0]), delimiter=',')
        file_targets = cnt_pd[cnt_pd.columns[0]].values[:-1]
        file_counts = cnt_pd[cnt_pd.columns[1]].values[:-1]
    folder_counts = np.zeros((len(targets),), dtype=int)
    for i, t in enumerate(targets):
        source_path = os.path.join(data_path, t)
        samples = os.listdir(source_path)
        folder_counts[i] = len(samples)
        if csv_file_exist:
            img_counts_matches = img_counts_matches and (folder_counts[i] == file_counts[i])
            assert (file_targets[i] == t), "{:s}!={:s}".format(file_targets[i], t)

    return img_counts_matches, targets, folder_counts

def create_dataset(path_dict, user_id_dict, params_dict):
    data_path = path_dict["data_base"]  # original path of data to load
    train_path = path_dict["train"]  # train data to create
    valid_path = path_dict["valid"]  # valid data to create
    test_path = path_dict["test"]  # test data to create
    cnt_table_fileName = os.path.join(os.path.abspath(os.path.join(path_dict["train"], os.pardir)), "cnt_table" +
                                      params_dict["exp_ident"] + ".csv")

    img_cnt_ok_all, targets, cnt_vec_all = validate_downloaded_data(data_path)
    if not img_cnt_ok_all:
        print("download the data again!!")
        sys.exit(21)

    table_rows = targets.copy()
    table_rows.append("total")
    cnt_table = pd.DataFrame(index=table_rows, columns=["train", "validation", "test", "total"])
    for col in cnt_table.columns:
        cnt_table[col].values[:] = 0

    if os.path.isdir(train_path) and os.path.isdir(valid_path) and os.path.isdir(test_path):
        try:
            targets_tr, img_cnt_tr = count_data_in_folder(train_path)
            cnt_table["train"].values[:-1] = img_cnt_tr
            targets_va, img_cnt_va = count_data_in_folder(valid_path)
            cnt_table["validation"].values[:-1] = img_cnt_va
            targets_te, img_cnt_te = count_data_in_folder(test_path)
            cnt_table["test"].values[:-1] = img_cnt_te
            cnt_table["total"].values[:-1] = img_cnt_tr + img_cnt_va + img_cnt_te
            cnt_table[-1:].values[:] = np.sum(cnt_table[:-1].values[:], axis=0)
            if np.sum(cnt_vec_all - img_cnt_tr - img_cnt_va - img_cnt_te)==0:
                return cnt_table
            else:
                rmtree(train_path, ignore_errors=True)
                rmtree(valid_path, ignore_errors=True)
                rmtree(test_path, ignore_errors=True)
        except:
            rmtree(train_path, ignore_errors=True)
            rmtree(valid_path, ignore_errors=True)
            rmtree(test_path, ignore_errors=True)

    create_sub_folders(targets, train_path)
    create_sub_folders(targets, valid_path)
    create_sub_folders(targets, test_path)
    for col in cnt_table.columns:
        cnt_table[col].values[:] = 0

    np.random.seed(seed=params_dict["randomSeed"])
    spaces_list = []
    for t in targets:
        print(f"Start copying target {t} -->")
        source_path = os.path.join(data_path, t)
        samples = os.listdir(source_path)
        #according to user_id_dict
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
            #if user_id_dict["valid"] == user_id_int:
            #    copyfile(os.path.join(source_path, s), os.path.join(valid_path, t, s))
            #    cnt_table["validation"][t] += 1
            if user_id_dict["test"] == user_id_int:
                copyfile(os.path.join(source_path, s), os.path.join(test_path, t, s))
                cnt_table["test"][t] += 1
            else:
                copyfile(os.path.join(source_path, s), os.path.join(train_path, t, s))
                train_samples.append(os.path.join(train_path, t, s))
                cnt_table["train"][t] += 1
        # deal with validation samples
        num_of_train_samples = len(train_samples)
        perm_list = np.random.permutation(num_of_train_samples)
        spaces = np.array(np.floor(np.linspace(0.0, num_of_train_samples, num=6)), dtype=int)
        fr, to = spaces[user_id_dict["cross_valid_id"]-1], spaces[user_id_dict["cross_valid_id"]]
        spaces_list.append(list(np.array([fr, to])) + list([-1])+ list(perm_list[fr:to]))
        for i in range(fr, to):
            sample_to_move = train_samples[perm_list[i]]
            sample_new_name = sample_to_move.replace(train_path, valid_path)
            os.rename(sample_to_move, sample_new_name)
            cnt_table["train"][t] -= 1
            cnt_table["validation"][t] += 1

        cnt_table["train"]["total"] += cnt_table["train"][t]
        cnt_table["validation"]["total"] += cnt_table["validation"][t]
        cnt_table["test"]["total"] += cnt_table["test"][t]
        print(f"Copied {t} --> train({cnt_table['train'][t]}),valid,({cnt_table['validation'][t]})test({cnt_table['test'][t]})")

    pd.DataFrame.to_csv(cnt_table, path_or_buf=cnt_table_fileName)
    print('\n'.join(map(str, spaces_list)))
    samples_list_filename = cnt_table_fileName.replace(".csv", "_sl.txt")
    with open(samples_list_filename, 'w') as f:
        for i, item in enumerate(spaces_list):
            f.write("%s - %s\n" % (str(targets[i]), str(item)))

    return cnt_table

def get_create_folders(params_dict):
    data_path_base = params_dict["data_path_base"]

    data_ident = 'data' + params_dict["exp_ident"]
    base_dir = funcH.getVariableByComputerName('base_dir')  # xx/DataPath or xx/DataFolder
    results_dir = os.path.join(base_dir, 'sup', 'results')
    models_dir = os.path.join(base_dir, 'sup', 'models')
    exp_ident_str = 'rs' + str(params_dict["randomSeed"]).zfill(2)
    data_path_fill = os.path.join(base_dir, 'sup', data_ident)
    data_path_train = os.path.join(data_path_fill, data_path_base + '_' + exp_ident_str + '_tr')
    data_path_valid = os.path.join(data_path_fill, data_path_base + '_' + exp_ident_str + '_va')
    data_path_test = os.path.join(data_path_fill, data_path_base + '_' + exp_ident_str + '_te')
    data_path_base = os.path.join(base_dir, data_path_base, "imgs")
    result_fold = os.path.join(base_dir, 'sup', 'pred' + params_dict["exp_ident"])

    path_dict = {
        "results": results_dir,
        "models": models_dir,
        "data_base": data_path_base,  # original path of data to load
        "train": data_path_train,  # train data to create
        "valid": data_path_valid,  # valid data to create
        "test": data_path_test,  # test data to create
        "result_fold": result_fold,  # to save the predictions and labels
    }

    funcH.createDirIfNotExist(results_dir)
    funcH.createDirIfNotExist(models_dir)
    funcH.createDirIfNotExist(data_path_train)
    funcH.createDirIfNotExist(data_path_valid)
    funcH.createDirIfNotExist(data_path_test)
    funcH.createDirIfNotExist(result_fold)

    return path_dict

def create_exp_name(params_dict):
    expName = params_dict["exp_ident"] + \
              '_rs' + str(params_dict["randomSeed"]).zfill(2)
    return expName

def get_dataset_variables(path_dict, train_data_transform, valid_data_transform, batch_size, num_workers):
    train_tr = khs_dataset(root_dir=path_dict["train"], transform=train_data_transform,
                                datasetname='user_independent_khs')
    train_va = khs_dataset(root_dir=path_dict["train"], transform=valid_data_transform,
                                datasetname='user_independent_khs')
    val_dataset = khs_dataset(root_dir=path_dict["valid"], transform=valid_data_transform,
                              datasetname='user_independent_khs')
    test_dataset = khs_dataset(root_dir=path_dict["test"], transform=valid_data_transform,
                               datasetname='user_independent_khs')
    ds = {
        "train_tr": torch.utils.data.DataLoader(train_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "train_te": torch.utils.data.DataLoader(train_va, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "valid": torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    train_labels = train_va.labels
    num_classes = np.unique(train_labels).size
    return ds, train_labels, num_classes

def main(argv):
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    params_dict, user_id_dict = parseArgs(argv)
    path_dict = get_create_folders(params_dict)
    cnt_table = create_dataset(path_dict, user_id_dict, params_dict)

    print('you are running this train function on = <', params_dict["hostName"], '>')

    input_initial_resize, input_size, batch_size, num_workers = initSomeVals(params_dict)
    train_data_transform, valid_data_transform = getTransformFuncs(input_size, input_initial_resize)

    expName = create_exp_name(params_dict)
    result_csv_file = os.path.join(path_dict["results"], 'rCF' + expName + '.csv')

    epochFr, epochTo = setEpochBounds(result_csv_file, params_dict["epochs"], params_dict["appendEpochBinary"])
    print("epochFr({:d}), epochTo({:d})".format(epochFr, epochTo), flush=True)

    ds_loader, train_labels, num_classes = get_dataset_variables(path_dict, train_data_transform, valid_data_transform, batch_size, num_workers)
    print(cnt_table)

    model, optimizer, updatedModelFile = getModel(params_dict, path_dict, expName, num_classes)

    num_ftrs = model.fc.in_features
    print('num_classes = ', num_classes, ', num_ftrs = ', num_ftrs, flush=True)

    epochStartTime = time.time()
    result_row = iterate_1(model, ds_loader, num_ftrs, epochFr-1, epochTo, epochStartTime, path_dict)

    if not os.path.isfile(result_csv_file):
        np.savetxt(result_csv_file, np.array(result_row).reshape(1, -1), fmt='%4.3f', delimiter='*', newline=os.linesep,
               header='ep * acc_tra * acc_val * acc_tes', footer='', comments='', encoding=None)
    else:
        result_pd = pd.read_csv(result_csv_file, header=0, sep="*", names=["epoch", "train", "validation", "test"])
        ep_read = result_pd["epoch"].values[-1]
        tr_acc_rd = result_pd["train"].values[-1]
        va_acc_rd = result_pd["validation"].values[-1]
        te_acc_rd = result_pd["test"].values[-1]
        good_to_proceed = (epochFr==ep_read+1) and (tr_acc_rd==result_row[1]) and (va_acc_rd==result_row[2]) and (te_acc_rd==result_row[3])
        if not good_to_proceed:
            print("result_row=", result_row)
            print("final row=", result_pd.iloc[[-1]])
            saveToResultMatFile(result_csv_file, result_row)


    for ep in range(epochFr, epochTo):
        model.train()  # Set model to training mode
        epochStartTime = time.time()
        _, _ = runTrainDs(model, optimizer, ds_loader["train_tr"])

        result_row = iterate_1(model, ds_loader, num_ftrs, ep, epochTo, epochStartTime, path_dict)
        saveToResultMatFile(result_csv_file, result_row)
        torch.save(model, f=updatedModelFile)

if __name__ == '__main__':
    main(sys.argv)