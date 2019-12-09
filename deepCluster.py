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
from dataset import HandShapeDataset

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

    # feature_cls_path = os.path.join(self.opts.feature_path, cls)
    # if not os.path.exists(feature_cls_path):
    #     os.makedirs(feature_cls_path)
    #
    # np.savez(os.path.join(feature_cls_path, video + '.npz'), np.array(features))
    return features

def runTrainDs(model, optimizer, dsLoad_train_train):
    print("running --> runTrainDs", datetime.datetime.now().strftime("%H:%M:%S"))
    t = time.time()
    running_acc = []
    idShuffle = []
    cnt = 0
    for sample in dsLoad_train_train:
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
        acc = ((indices == labels).cpu().numpy().astype(dtype=np.float))
        for x in acc:
            running_acc.append(x)
        idShuffle = idShuffle + ids.tolist()

    tr_acc_run = (np.sum(np.array(running_acc)) / len(np.array(running_acc))).astype(dtype=np.float)
    elapsed = time.time() - t

    funcH.removeLastLine()
    print('runTrainDs completed (', funcH.getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))

    return tr_acc_run, idShuffle

def runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=512):
    cnt = 0
    idSorted = []
    epoc_acc = []

    features_avgPool = []
    labels_avgPool = []
    predictions_avgPool = []

    print("running --> runValidDs(return_feats=", str(return_feats), ", layerSize=", str(layerSize), ")", datetime.datetime.now().strftime("%H:%M:%S"))
    t = time.time()

    for sample in dsLoad_train_featExtract:
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        ids = sample['id'].cuda()
        outputs = model(images)

        _, indices = outputs.max(1)
        acc = ((indices == labels).cpu().numpy().astype(dtype=np.float))
        for x in acc:
            epoc_acc.append(x)

        cnt += 1
        if return_feats:
            feats = extract_features(layerSize=layerSize, model=model, feature_layer_string='avgpool', images=images)
            features_avgPool = features_avgPool + feats
            labels_avgPool = labels_avgPool + labels.tolist()
            predictions_avgPool = predictions_avgPool + indices.tolist()
        idSorted = idSorted + ids.tolist()

    elapsed = time.time() - t
    val_acc_epoch = (np.sum(np.array(epoc_acc)) / len(np.array(epoc_acc))).astype(dtype=np.float)

    funcH.removeLastLine()
    print('runValidDs(return_feats=', str(return_feats), ' completed (', funcH.getElapsedTimeFormatted(elapsed), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))

    return val_acc_epoch, idSorted, features_avgPool, labels_avgPool, predictions_avgPool

def parseArgs(argv):
    # deepCluster.py --modelName resnet18 --posterior_dim 256 --numOfSigns 11 --epochs 50
    #                --batch_size 16 --appendEpochBinary 1 --randomSeed 1 --clusterModel KMeans
    param01 = {"paramName": "modelName", "possibleValues": "{'resnet18','vgg16','rsa','corsa'}",
               "vs": "-mn", "defaultVal": "resnet18", "dvSet": True, "paramType": "str"}
    param02 = {"paramName": "use_updated_model", "possibleValues": "{True or False",
               "vs": "-ua", "defaultVal": 1, "dvSet": True, "paramType": "bool"}
    param03 = {"paramName": "posterior_dim", "possibleValues": "{32,64,128,256}",
               "vs": "-pd", "defaultVal": 256, "dvSet": True, "paramType": "int"}
    param04 = {"paramName": "numOfSigns", "possibleValues": "{11, 41}",
               "vs": "-ns", "defaultVal": 11, "dvSet": True, "paramType": "int"}
    param05 = {"paramName": "epochs", "possibleValues": "{50,200,500}",
               "vs": "-ep", "defaultVal": 50, "dvSet": True, "paramType": "int"}
    param06 = {"paramName": "batch_size", "possibleValues": "{0-B}",
               "vs": "-bs", "defaultVal": 16, "dvSet": True, "paramType": "int"}
    param07 = {"paramName": "appendEpochBinary", "possibleValues": "{0, 1}",
               "vs": "-ea", "defaultVal": 0, "dvSet": True, "paramType": "int"}
    param08 = {"paramName": "randomSeed", "possibleValues": "{some integer}",
               "vs": "-rs", "defaultVal": 1, "dvSet": True, "paramType": "int"}
    param09 = {"paramName": "clusterModel", "possibleValues": "{'KMeans','GMM_diag','Spectral'}",
               "vs": "-cm", "defaultVal": "KMeans", "dvSet": True, "paramType": "str"}
    param10 = {"paramName": "initialLabel", "possibleValues": "{None,'baseResults_featName_pcaCnt_baseClusterModel','fileName_<relativePathUnderResults>'}",
               "vs": "-il", "defaultVal": None, "dvSet": True, "paramType": "str"}
    param11 = {"paramName": "clusterLabelUpdateInterval", "possibleValues": "{some integer}",
               "vs": "-cli", "defaultVal": 2, "dvSet": True, "paramType": "int"}

    # model and data parameters
    paramsAll = [param01, param02, param03, param04, param05,
                 param06, param07, param08, param09, param10, param11]

    argSetDescriptions = ""
    go_00 = "hi:o:"
    go_01 = ["help="]

    excepStr = "aeCluster.py "
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

    valuesParamsCur = {}
    dvSetParamsCur = {}
    for i in range(len(paramsAll)):
        valuesParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["defaultVal"]
        dvSetParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["dvSet"]

    def setByParamName(paramsAll,curParamName,valuesParamsCur,dvSetParamsCur):
        for i in range(len(paramsAll)):
            if paramsAll[i]["paramName"]==curParamName:
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

    params_dict = {
        "modelName": valuesParamsCur["modelName"],
        "use_updated_model": valuesParamsCur["use_updated_model"],
        "posterior_dim": valuesParamsCur["posterior_dim"],
        "numOfSigns": valuesParamsCur["numOfSigns"],
        "clusterModel": valuesParamsCur["clusterModel"],
        "epochs": valuesParamsCur["epochs"],
        "batch_size": valuesParamsCur["batch_size"],
        "randomSeed": valuesParamsCur["randomSeed"],
        "appendEpochBinary": valuesParamsCur["appendEpochBinary"],
        "initialLabel": valuesParamsCur["initialLabel"],
        "clusterLabelUpdateInterval": valuesParamsCur["clusterLabelUpdateInterval"],
    }
    return params_dict

def initSomeVals(params_dict):
    useModelName = params_dict["modelName"]
    batch_size = params_dict["batch_size"]
    if useModelName == 'resnet18' or useModelName == 'resnet34' or useModelName == 'resnet50' or useModelName == 'resnet101':
        input_initial_resize = 256
        input_size = 224
        batch_size = 15
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

def calc_stats_on_iterate(featTrInit, labelsTrInit, predictionsTr, k, clusterModel):
    nmi_lab, predClusters, nmi_lab_nonzero = funcH.get_nmi_deepCluster(featTrInit, labelsTrInit, k, clusterModel=clusterModel)
    acc_lab, _ = funcH.accFromKlusterLabels(labelsTrInit, predClusters, removeZeroLabels=False)
    acc_lab_nonzero, _ = funcH.accFromKlusterLabels(labelsTrInit, predClusters, removeZeroLabels=True)
    nmi_pred = funcH.get_nmi_only(predictionsTr, predClusters, average_method='geometric')
    x, y = funcH.getNonZeroLabels(predictionsTr, predClusters)
    nmi_pred_nonzero = funcH.get_nmi_only(x, y, average_method='geometric')
    acc_pred, _ = funcH.accFromKlusterLabels(predictionsTr, predClusters, removeZeroLabels=False)
    acc_pred_nonzero, _ = funcH.accFromKlusterLabels(predictionsTr, predClusters, removeZeroLabels=True)

    return nmi_lab, acc_lab, nmi_lab_nonzero, acc_lab_nonzero, \
           nmi_pred, acc_pred, nmi_pred_nonzero, acc_pred_nonzero, predClusters

def iterate_1(featTrInit, labelsTrInit, predictionsTr, k, labelSaveFileName, ep, epochTo, trAccInit, epochStartTime,
              clusterModel='KMeans'):
    labelsTrInit = np.asarray(labelsTrInit, dtype=int)
    predictionsTr = np.asarray(predictionsTr, dtype=int)

    nmi_lab, acc_lab, nmi_lab_nz, acc_lab_nz, \
    nmi_pred, acc_pred, nmi_pred_nz, acc_pred_nz, predClusters = \
                                                        calc_stats_on_iterate(featTrInit, labelsTrInit, predictionsTr,
                                                                              k, clusterModel)

    np.savez(labelSaveFileName, labelsTrInit=labelsTrInit, predClusters=predClusters, acc_lab=acc_lab, acc_lab_nonzero=acc_pred_nz, predictionsTr=predictionsTr)

    resultRow = np.array([ep, trAccInit, nmi_lab, nmi_lab_nz, acc_lab, acc_lab_nz, nmi_pred, nmi_pred_nz, acc_pred, acc_pred_nz])
    print('ep={:d}/{:d}, trAccInit={:0.5f} - '
          'nmi_lab={:0.2f}, nmi_lab_nonzero={:0.2f}, acc_lab={:0.2f}, acc_lab_nonzero={:0.2f}, '
          'nmi_pred={:0.2f}, nmi_pred_nonzero={:0.2f}, acc_pred={:0.2f}, acc_pred_nonzero={:0.2f} '.format(
            ep, epochTo, trAccInit, nmi_lab, nmi_lab_nz, acc_lab, acc_lab_nz, nmi_pred, nmi_pred_nz, acc_pred, acc_pred_nz))
    #elapsed time of epoch
    print('Epoch done in (', funcH.getElapsedTimeFormatted(time.time() - epochStartTime), '), ended at ', datetime.datetime.now().strftime("%H:%M:%S"))
    print('*-*-*-*-*-*-*')

    return predClusters, resultRow

def saveToResultMatFile(resultMatFile, resultRow):
    f = open(resultMatFile, 'a')
    np.savetxt(f, np.array(resultRow).reshape(1, -1), fmt='%4.3f', delimiter='*', newline=os.linesep, header='', footer='', comments='', encoding=None)
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
        transforms.Resize(input_initial_resize),
        transforms.RandomSizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_data_transform, valid_data_transform

def setEpochBounds(labelSaveFolder, num_epochs, appendEpochBinary):
    completedEpochCnt = funcH.numOfFilesInFolder(labelSaveFolder, startswith="labels", endswith=".npz")
    epochFr = completedEpochCnt
    epochTo = num_epochs + appendEpochBinary*completedEpochCnt
    if epochFr >= epochTo:
        print('epochFr(', str(epochFr), ') >= (', str(epochTo), ')epochTo. hence exiting.')
        sys.exit(5)
    print('epochFr = ', str(epochFr), ', epochTo = ', str(epochTo))
    return epochFr, epochTo

def getModel(params_dict, modelsDir, expName):

    useModelName = params_dict["modelName"]  # 'resnet18'
    useUpdatedModel = params_dict["use_updated_model"]  # True
    downloadedModelFile = os.path.join(modelsDir, 'model_' + useModelName + '_Down.model')
    updatedModelFile = os.path.join(modelsDir, 'model_' + expName + '.model')

    if os.path.isfile(updatedModelFile) and useUpdatedModel:
        print('model(', useModelName, ') is being loaded from updatedModelFile')
        model = torch.load(f=updatedModelFile)
        print('model(', useModelName, ') has been loaded from updatedModelFile')
    elif os.path.isfile(downloadedModelFile):
        print('model(', useModelName, ') is being loaded from downloadedModelFile')
        model = torch.load(f=downloadedModelFile)
        print('model(', useModelName, ') has been loaded from downloadedModelFile')
    elif useModelName == 'resnet18' or useModelName == 'resnet34' or useModelName == 'resnet50' or useModelName == 'resnet101':
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
    elif useModelName == 'vgg16':
        model = models.vgg16(pretrained=True)
        print('model(', useModelName, ') has been downloaded and being saved to ', downloadedModelFile)
        torch.save(model, f=downloadedModelFile)

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

def decode_initial_label_param(initialLabelParam):

    if initialLabelParam is None:
        initialLabelVec = None
    else:
        initialLabelVecStrings = initialLabelParam.split("_")
        if initialLabelVecStrings[0] == 'fn':
            fileName_end = initialLabelVecStrings[1]  # 'baseResults/hgsk256_11_KMeans_256.npz'
            results_dir = funcH.getVariableByComputerName('results_dir').replace("bdResults", "dcResults")
            labelFileFullName = os.path.join(results_dir, fileName_end)
            #  np.savez(predictionFileNameFull, labels_all, predClusters)
            npzDict = np.load(labelFileFullName, allow_pickle=True)
            initialLabelVec = npzDict["arr_1"]
        else:
            print('Not implemented yet')
            os._exit(30)
        if initialLabelVecStrings[0] == "baseResults":
            print('Not implemented yet')
            os._exit(30)

    return initialLabelVec

def updateTrainLabels(train_dataset, clusterLabelUpdateInterval, epochID, predClusters=None, initialLabelVec=None):
    modVal = np.mod(epochID, clusterLabelUpdateInterval)
    if initialLabelVec is not None and epochID < clusterLabelUpdateInterval:
        print("updating  train labels with initialLabelVec at first epochID(", str(epochID), ")")
        train_dataset.updateLabels(list(initialLabelVec))
    elif modVal == 0:
        print("updating  train labels at epochID(", str(epochID))
        train_dataset.updateLabels(list(predClusters))
    else:
        print("epochID(", str(epochID), " mod  ", str(clusterLabelUpdateInterval) ,") clusterLabelUpdateInterval = ", str(modVal) ,".. not updating train labels")
    return train_dataset

def main(argv):
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

    params_dict = parseArgs(argv)
    numOfSigns = params_dict["numOfSigns"]  # 11 or 41
    clusterModel = params_dict["clusterModel"]  # 'KMeans', 'GMM_diag', 'Spectral'
    params_dict["hostName"] = socket.gethostname()
    initialLabelVec = decode_initial_label_param(params_dict["initialLabel"])
    clusterLabelUpdateInterval = params_dict["clusterLabelUpdateInterval"]

    print('you are running this train function on = <', params_dict["hostName"], '>')

    input_initial_resize, input_size, batch_size, num_workers = initSomeVals(params_dict)
    train_data_transform, valid_data_transform = getTransformFuncs(input_size, input_initial_resize)

    base_dir = funcH.getVariableByComputerName('base_dir')  # dataPath and dataFolder
    data_dir = funcH.getVariableByComputerName('data_dir')  # bdData
    results_dir = funcH.getVariableByComputerName('results_dir').replace("bdResults", "dcResults")
    labelsDir = funcH.getVariableByComputerName('results_dir').replace("bdResults", "dcLabels")
    modelsDir = os.path.join(base_dir, 'dcModels')
    nnVidsDir = os.path.join(base_dir, 'neuralNetHandVideos_' + str(numOfSigns))

    expName = params_dict["modelName"] + '_' + \
              params_dict["clusterModel"] + \
              '_pd' + str(params_dict["posterior_dim"]) + \
              '_' + str(numOfSigns)
    labelSaveFolder = os.path.join(labelsDir, expName)
    resultMatFile = os.path.join(results_dir, 'rMF_' + expName)

    funcH.createDirIfNotExist(results_dir)
    funcH.createDirIfNotExist(labelsDir)
    funcH.createDirIfNotExist(modelsDir)
    funcH.createDirIfNotExist(labelSaveFolder)

    epochFr, epochTo = setEpochBounds(labelSaveFolder, params_dict["epochs"], params_dict["appendEpochBinary"])

    train_dataset = HandShapeDataset(root_dir=nnVidsDir, istrain=True, transform=train_data_transform, datasetname='nnv')
    val_dataset = HandShapeDataset(root_dir=nnVidsDir, istrain=False, transform=valid_data_transform, datasetname='nnv')

    num_classes = np.unique(train_dataset.labels).size

    print('trainCnt = ', len(train_dataset))
    print('valCnt = ', len(val_dataset))

    model, optimizer, updatedModelFile = getModel(params_dict, modelsDir, expName)

    num_ftrs = model.fc.in_features
    print('num_classes = ', num_classes, ', num_ftrs = ', num_ftrs, flush=True)

    epochStartTime = time.time()

    dsLoad_train_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dsLoad_train_featExtract = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()

    #  evaluate the model to extract
    #  trAccInit : to save as initial training accuracy
    #  featTrInit : features to cluster, also saved as result features in -saveFeatsExtracted-
    #  labelsTrInit :
    #  predictionsTrInit :
    trAccInit, _, featTrInit, labelsTrInit, predictionsTrInit = runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=num_ftrs)

    saveFeatsExtracted(data_dir, epochFr, params_dict["modelName"], expName, featTrInit, labelsTrInit, predictionsTrInit)

    labelSaveFileName = labelSaveFolder + os.sep + 'labels_{:03d}.npz'.format(epochFr)
    predClusters, resultRow = iterate_1(featTrInit, labelsTrInit, predictionsTrInit, params_dict["posterior_dim"],
                                        labelSaveFileName, epochFr-1, epochTo, trAccInit,
                                        epochStartTime, clusterModel=clusterModel)

    train_dataset = updateTrainLabels(train_dataset, clusterLabelUpdateInterval, epochFr, predClusters=predClusters, initialLabelVec=initialLabelVec)

    resultMat = []
    resultMat = resultMat + resultRow.tolist()
    if not os.path.isfile(resultMatFile):
        np.savetxt(resultMatFile, np.array(resultRow).reshape(1, -1), fmt='%4.3f', delimiter='*', newline=os.linesep,
               header='ep * tr_acc_epoch * nmi_lab * nmi_lab_nz * acc_lab * acc_lab_nz * nmi_pred * nmi_pred_nz * acc_pred * acc_pred_nz',
               footer='', comments='', encoding=None)
    else:
        f = open(resultMatFile, 'a')
        np.savetxt(f, np.array(resultRow).reshape(1, -1), fmt='%4.3f', delimiter='*', newline=os.linesep, header='', footer='', comments='', encoding=None)
        f.close()


    for ep in range(epochFr, epochTo):
        model.train()  # Set model to training mode
        epochStartTime = time.time()
        _, _ = runTrainDs(model, optimizer, dsLoad_train_train)

        model.eval()
        tr_acc_epoch, _, features_avgPool, labels_avgPool, predictionsTr = \
            runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=num_ftrs)

        labelSaveFileName = labelSaveFolder + os.sep + 'labels_{:03d}.npz'.format(ep+1)
        predClusters, resultRow = iterate_1(features_avgPool, labelsTrInit, predictionsTr, params_dict["posterior_dim"], labelSaveFileName, ep, epochTo, tr_acc_epoch, epochStartTime, clusterModel=clusterModel)
        resultMat = resultMat + resultRow.tolist()

        train_dataset = updateTrainLabels(train_dataset, clusterLabelUpdateInterval, ep, predClusters=predClusters)

        saveFeatsExtracted(data_dir, ep, params_dict["modelName"], expName, features_avgPool, labelsTrInit, predictionsTr)
        saveToResultMatFile(resultMatFile, resultRow)
        torch.save(model, f=updatedModelFile)

if __name__ == '__main__':
    main(sys.argv)