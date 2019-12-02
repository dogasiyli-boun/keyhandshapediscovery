from deepClusterPaper.dataset import HandShapeDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms, utils
import torchvision.models as models
import os
import socket
import helperFuncs as funcH
import getopt
import sys

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

def runTrainDs(model, dsLoad_train_train):
    accuracy = 0
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
    return tr_acc_run, idShuffle

def runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=512):
    cnt = 0
    idSorted = []
    epoc_acc = []

    features_avgPool = []
    labels_avgPool = []
    predictions_avgPool = []

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

    val_acc_epoch = (np.sum(np.array(epoc_acc)) / len(np.array(epoc_acc))).astype(dtype=np.float)

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

    # model and data parameters
    paramsAll = [param01, param02, param03, param04, param05,
                 param06, param07, param08, param09]

    argSetDescriptions = ""
    go_00 = "hi:o:"
    go_01 = ["help="]

    excepStr = "clusterDeep.py "
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

def iterate_1(featTrInit, labelsTrInit, predictionsTr, k, labelSaveFileName, ep, num_epochs, trAccInit,
              clusterModel='KMeans'):
    labelsTrInit = np.asarray(labelsTrInit, dtype=int)
    predictionsTr = np.asarray(predictionsTr, dtype=int)

    nmi_lab, predClusters, nmi_lab_nonzero = funcH.get_nmi_deepCluster(featTrInit, labelsTrInit, k,
                                                                       clusterModel=clusterModel)
    acc_lab, _ = funcH.accFromKlusterLabels(labelsTrInit, predClusters, removeZeroLabels=False)
    acc_lab_nonzero, _ = funcH.accFromKlusterLabels(labelsTrInit, predClusters, removeZeroLabels=True)

    nmi_pred = funcH.get_nmi_only(predictionsTr, predClusters, average_method='geometric')
    x, y = funcH.getNonZeroLabels(predictionsTr, predClusters)
    nmi_pred_nonzero = funcH.get_nmi_only(x, y, average_method='geometric')
    acc_pred, _ = funcH.accFromKlusterLabels(predictionsTr, predClusters, removeZeroLabels=False)
    acc_pred_nonzero, _ = funcH.accFromKlusterLabels(predictionsTr, predClusters, removeZeroLabels=True)

    np.savez(labelSaveFileName, labelsTrInit=labelsTrInit, predClusters=predClusters, acc_lab=acc_lab, acc_lab_nonzero=acc_lab_nonzero, predictionsTr=predictionsTr)

    resultRow = np.array([ep, trAccInit, nmi_lab, nmi_lab_nonzero, acc_lab, acc_lab_nonzero, nmi_pred, nmi_pred_nonzero, acc_pred, acc_pred_nonzero])
    print('ep={:d}/{:d}, trAccInit={:0.5f} - '
          'nmi_lab={:0.2f}, nmi_lab_nonzero={:0.2f}, acc_lab={:0.2f}, acc_lab_nonzero={:0.2f}, '
          'nmi_pred={:0.2f}, nmi_pred_nonzero={:0.2f}, acc_pred={:0.2f}, acc_pred_nonzero={:0.2f} '.format(
            ep, num_epochs, trAccInit, nmi_lab, nmi_lab_nonzero, acc_lab, acc_lab_nonzero, nmi_pred, nmi_pred_nonzero, acc_pred, acc_pred_nonzero))

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

def main(argv):
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

    params_dict = parseArgs(argv)
    numOfSigns = params_dict["numOfSigns"]  # 11 or 41
    clusterModel = params_dict["clusterModel"]  # 'KMeans', 'GMM_diag', 'Spectral'
    params_dict["hostName"] = socket.gethostname()

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

    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=num_workers)
    num_classes = np.unique(train_dataset.labels).size

    print('trainCnt = ', len(train_dataset))
    print('valCnt = ', len(val_dataset))

    model, optimizer, updatedModelFile = getModel(params_dict, modelsDir, expName)

    num_ftrs = model.fc.in_features
    print('num_classes = ', num_classes, ', num_ftrs = ', num_ftrs, flush=True)

    model.eval()
    dsLoad_train_featExtract = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trAccInit, idTrInit, featTrInit, labelsTrInit, predictionsTrInit = runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=num_ftrs)

    saveFeatsExtracted(data_dir, epochFr, params_dict["modelName"], expName, featTrInit, labelsTrInit, predictionsTrInit)

    labelSaveFileName = labelSaveFolder + os.sep + 'labels_{:03d}.npz'.format(epochTo)
    predClusters, resultRow = iterate_1(featTrInit, labelsTrInit, predictionsTrInit, params_dict["posterior_dim"], labelSaveFileName, epochFr-1, epochTo, trAccInit, clusterModel=clusterModel)
    train_dataset.updateLabels(list(predClusters))

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
        dsLoad_train_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        tr_acc_run, idTrShuffle = runTrainDs(model, dsLoad_train_train)

        model.eval()
        val_acc_epoch, idValSorted, _, _, _ = runValidDs(model, val_dataset_loader, return_feats=False, layerSize=num_ftrs)
        dsLoad_train_featExtract = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        tr_acc_epoch, idTrSorted, features_avgPool, labels_avgPool, predictionsTr = runValidDs(model, dsLoad_train_featExtract, return_feats=True, layerSize=num_ftrs)

        #predLabs_?x --> has the previous
        labelSaveFileName = labelSaveFolder + os.sep + 'labels_{:03d}.npz'.format(ep+1)
        predClusters, resultRow = iterate_1(features_avgPool, labelsTrInit, predictionsTr, params_dict["posterior_dim"], labelSaveFileName, ep, num_epochs, tr_acc_epoch, clusterModel=clusterModel)
        resultMat = resultMat + resultRow.tolist()
        train_dataset.updateLabels(list(predClusters))

        saveFeatsExtracted(data_dir, ep, params_dict["modelName"], expName, features_avgPool, labelsTrInit, predictionsTr)
        saveToResultMatFile(resultMatFile, resultRow)
        torch.save(model, f=updatedModelFile)

if __name__ == '__main__':
    main(sys.argv)