''' Code for Keyframe Detection,
	Authors : Doga Siyli and Batuhan Gundogdu
'''
# -*- coding: iso-8859-15 -*-

from sys import argv
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K
#%%
import modelFuncs as funcM
import helperFuncs as funcH
import dataLoaderFuncs as funcD
import trainLoops as funcTL
import projRelatedHelperFuncs as funcPRH

import sys
import getopt

from numpy.random import seed

def parseArgs(argv):
    #aeCluster.py --trainMode rsa
    # --trainMode sae --posterior_dim 64
    # --trainMode cosae --posterior_dim 128
    # --trainMode rsa --posterior_dim 256
    param01 = {"paramName": "trainMode", "possibleValues": "{'sae','cosae','rsa','corsa'}",
               "vs": "-mt", "defaultVal": "rsa", "dvSet": True, "paramType": "str"}
    param02 = {"paramName": "posterior_dim", "possibleValues": "{32,64,128,256}",
               "vs": "-pd", "defaultVal": 256, "dvSet": True, "paramType": "int"}
    param03 = {"paramName": "weight_of_regularizer", "possibleValues": "{0.2,0.5,1.0}",
               "vs": "-wr", "defaultVal": 1.0, "dvSet": True, "paramType": "float"}
    param04 = {"paramName": "dataToUse", "possibleValues": "{'hog','resnet18','sn', 'skeleton'}",
               "vs": "-dt", "defaultVal": "hog", "dvSet": True, "paramType": "str"}
    param16 = {"paramName": "pcaCount", "possibleValues": "{-1, 32, 64, 96, 256, 512, 1024}",
               "vs": "-pc", "defaultVal": -1, "dvSet": True, "paramType": "int"}
    param17 = {"paramName": "numOfSigns", "possibleValues": "{11, 41}",
               "vs": "-ns", "defaultVal": 11, "dvSet": True, "paramType": "int"}
    param18 = {"paramName": "normMode", "possibleValues": "{'', 'nm', 'nl'}",
               "vs": "-nm", "defaultVal": '', "dvSet": True, "paramType": "string"}
    # model and data parameters
    params_model_data = [param01, param02, param03, param04, param16, param17, param18]

    # train parameters
    # --epochs 50 --appendEpochBinary 0 --batch_size 64 --applyCorr 0 --corr_randMode 0
    # --epochs 50 --appendEpochBinary 0 --batch_size 64 --applyCorr 2 --corr_randMode 0
    # --epochs 50 --appendEpochBinary 0 --batch_size 64 --applyCorr 2 --corr_randMode 1
    # --epochs 50 --appendEpochBinary 0 --batch_size 64 --applyCorr 2 --corr_randMode 0 --corr_swapMode 0
    param05 = {"paramName": "epochs", "possibleValues": "{50,200,500}",
               "vs": "-ep", "defaultVal": 50, "dvSet": True, "paramType": "int"}
    param06 = {"paramName": "corr_randMode", "possibleValues": "{0-False,1-True}",
               "vs": "-pd", "defaultVal": 0, "dvSet": True, "paramType": "bool"}
    param11 = {"paramName": "batch_size", "possibleValues": "{0-B}",
               "vs": "-bs", "defaultVal": 16, "dvSet": True, "paramType": "int"}
    param12 = {"paramName": "applyCorr", "possibleValues": "{0, 2}",
               "vs": "-wr", "defaultVal": 0, "dvSet": True, "paramType": "int"}
    param13 = {"paramName": "appendEpochBinary", "possibleValues": "{0, 1}",
               "vs": "-ea", "defaultVal": 0, "dvSet": True, "paramType": "int"}
    param15 = {"paramName": "randomSeed", "possibleValues": "{some integer}",
               "vs": "-rs", "defaultVal": 1, "dvSet": True, "paramType": "int"}
    param19 = {"paramName": "corr_swapMode", "possibleValues": "{0-False,1-True}",
               "vs": "-pd", "defaultVal": 1, "dvSet": True, "paramType": "bool"}
    paramsTrain = [param05, param06, param11, param12, param13, param15, param19]

    # rnn parameters
    # --rnnDataMode 0 --rnnTimesteps 10
    # --rnnDataMode 1 --rnnTimesteps 10 --rnnPatchFromEachVideo 10
    # --rnnDataMode 2 --rnnTimesteps 10 --rnnFrameOverlap 5
    param07 = {"paramName": "rnnDataMode", "possibleValues": "{0-lookBack,1-patchPerVideo,2-frameOverlap}",
                         "vs": "-rdm", "defaultVal": 0, "dvSet": True, "paramType": "int or str"}
    param08 = {"paramName": "rnnTimesteps", "possibleValues": "{1 to T}",
               "vs": "-rts", "defaultVal": -1, "dvSet": True, "paramType": "int"}
    param09 = {"paramName": "rnnPatchFromEachVideo", "possibleValues": "{1 to P}",
               "vs": "-rpv", "defaultVal": -1, "dvSet": True, "paramType": "int"}
    param10 = {"paramName": "rnnFrameOverlap", "possibleValues": "{1 to F}",
               "vs": "-rfo", "defaultVal": -1, "dvSet": True, "paramType": "int"}
    param14 = {"paramName": "rnnDropout", "possibleValues": "{0.5, 0.7}",
               "vs": "-rdo", "defaultVal": 0.5, "dvSet": True, "paramType": "float"}
    paramsRnn = [param07, param08, param09, param10, param14]

    paramsAll = params_model_data + paramsTrain + paramsRnn

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
                curParamName = param["paramName"]
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
                    elif param["paramType"] == "int or str":
                        if curParamName == "rnnDataMode":
                            rnnDataMode = str(arg)
                            if rnnDataMode == "lookBack" or rnnDataMode == "0":
                                param["defaultVal"] = 0
                            elif rnnDataMode == "patchPerVideo" or rnnDataMode == "1":
                                param["defaultVal"] = 1
                            elif rnnDataMode == "frameOverlap" or rnnDataMode == "2":
                                param["defaultVal"] = 2
                            else:
                                sys.exit(0)
                    else:
                        sys.exit(5)
                    param["dvSet"] = False

    valuesParamsCur = {}
    dvSetParamsCur = {}
    for i in range(len(paramsAll)):
        valuesParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["defaultVal"]
        dvSetParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["dvSet"]

    valuesParamsCur, dvSetParamsCur = funcM.initialize_RNN_Parameters(valuesParamsCur, dvSetParamsCur)

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

    defaultValsStr = " model and data parameters : "
    userSetValsStr = " model and data parameters : "
    for param in params_model_data:
        defaultValsStr, userSetValsStr = addValStr(defaultValsStr, userSetValsStr, param, ", ")

    defaultValsStr += os.linesep + "  train parameters : "
    userSetValsStr += os.linesep + "  train parameters : "
    for param in paramsTrain:
        defaultValsStr, userSetValsStr = addValStr(defaultValsStr, userSetValsStr, param, ", ")

    defaultValsStr += os.linesep + "  rnn parameters : "
    userSetValsStr += os.linesep + "  rnn parameters : "
    for param in paramsRnn:
        defaultValsStr, userSetValsStr = addValStr(defaultValsStr, userSetValsStr, param, ", ")

    print("Default values are set : ", os.linesep, defaultValsStr)
    print("Values set by user : ", os.linesep, userSetValsStr)

    for i in range(len(paramsAll)):
        valuesParamsCur[paramsAll[i]["paramName"]] = paramsAll[i]["defaultVal"]

    modelParams = {
        "trainMode": valuesParamsCur["trainMode"],
        "posterior_dim": valuesParamsCur["posterior_dim"],
        "weight_of_regularizer": valuesParamsCur["weight_of_regularizer"],
        "dataToUse": valuesParamsCur["dataToUse"],
        "pcaCount": valuesParamsCur["pcaCount"],
        "numOfSigns": valuesParamsCur["numOfSigns"],
        "normMode": valuesParamsCur["normMode"]
    }
    trainParams = {
        "epochs": valuesParamsCur["epochs"],
        "appendEpochBinary": valuesParamsCur["appendEpochBinary"],
        "batch_size": valuesParamsCur["batch_size"],
        "applyCorr": valuesParamsCur["applyCorr"],
        "corr_randMode": valuesParamsCur["corr_randMode"],
        "corr_swapMode": valuesParamsCur["corr_swapMode"],
        "randomSeed": valuesParamsCur["randomSeed"]
    }
    rnnParams = {
        "dataMode": valuesParamsCur["rnnDataMode"],
        "timesteps": valuesParamsCur["rnnTimesteps"],
        "patchFromEachVideo": valuesParamsCur["rnnPatchFromEachVideo"],
        "frameOverlap": valuesParamsCur["rnnFrameOverlap"],
        "dropout": valuesParamsCur["rnnDropout"]
    }
    return modelParams, trainParams, rnnParams

def getInitParams(trainParams, modelParams, rnnParams):
    subEpochs = 1

    if modelParams["trainMode"] == "sae":
        assert (trainParams["applyCorr"] == 0), "applyCorr(" + str(trainParams["applyCorr"]) + ") must be 0"
        trainParams["corr_randMode"] = 0
    if modelParams["trainMode"] == "cosae" and trainParams["applyCorr"] == 0:
        assert (trainParams["applyCorr"] >= 2), "applyCorr(" + str(trainParams["applyCorr"]) + ") must be >= 2"
    elif modelParams["trainMode"] == "rsa":
        if trainParams["applyCorr"] >= 2:
            modelParams["trainMode"] = "corsa"
            assert (rnnParams["dataMode"] == 0), "rnnDataMode(" + str(rnnParams["dataMode"]) + ") must be 0"
        trainParams["corr_randMode"] = 0
    elif modelParams["trainMode"] == "corsa":
        assert (rnnParams["dataMode"] == 0), "rnnDataMode(" + str(rnnParams["dataMode"]) + ") must be 0"
        assert (trainParams["applyCorr"] >= 2), "applyCorr(" + str(trainParams["applyCorr"]) + ") must be >= 2"

    exp_name = funcPRH.createExperimentName(trainParams=trainParams, modelParams=modelParams, rnnParams=rnnParams)

    return exp_name, subEpochs, trainParams, rnnParams

def initEpochIDsModelParams(trainFromScratch, trainParams, model, model_name, predictionLabelsDir):
    if not trainFromScratch and os.path.isfile(model_name):
        model.load_weights(model_name, by_name=True)
        predictedLabelsFileCount = len([f for f in os.listdir(predictionLabelsDir)
                                        if f.startswith('predicted_labels') and f.endswith('.npy') and os.path.isfile(
                os.path.join(predictionLabelsDir, f))])
        epochFr = predictedLabelsFileCount
        epochTo = trainParams["epochs"] + trainParams["appendEpochBinary"]*predictedLabelsFileCount
    else:
        epochFr = 0
        epochTo = trainParams["epochs"]

    print("model will run epoch from(", str(epochFr), ") to(", str(epochTo), ")")
    return model, epochFr, epochTo

## extra imports to set GPU options
################################### # TensorFlow wizardry 
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed 
config.gpu_options.allow_growth = True  
# Only allow a total of half the GPU memory to be allocated 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified. 
K.tensorflow_backend.set_session(tf.Session(config=config))

def main(argv):
    base_dir = funcH.getVariableByComputerName('base_dir')
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName('results_dir')
    print(argv)

    modelParams, trainParams, rnnParams = parseArgs(argv)

    seed(trainParams["randomSeed"])
    tf.set_random_seed(seed=trainParams["randomSeed"])

    numOfSigns = modelParams["numOfSigns"]
    feat_set, labels_all, detailed_labels_all = funcPRH.loadData(modelParams, numOfSigns, data_dir)
    data_dim = feat_set.shape[1]

    exp_name, subEpochs, trainParams, rnnParams = getInitParams(trainParams, modelParams, rnnParams)
    csv_name, model_name, outdir = funcPRH.createExperimentDirectories(results_dir, exp_name)
    model, modelTest, ES = funcM.getModels(data_dim=data_dim, modelParams=modelParams, rnnParams=rnnParams)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=False, period=1)
    csv_logger = CSVLogger(csv_name, append=True, separator=';')
    callbacks = [csv_logger, ES, checkpointer]

    #%%
    trainFromScratch = False
    epochCnt = trainParams["epochs"]
    predictionLabelsDir = results_dir + os.sep + 'results' + os.sep + exp_name
    model, epochFr, epochTo = initEpochIDsModelParams(trainFromScratch, trainParams, model, model_name, predictionLabelsDir)

    if epochFr == epochTo:
        print("+*-+*-+*-+*-epochs completed+*-+*-+*-+*-")
        exit(12)

    modelParams["callbacks"] = [csv_logger, checkpointer]
    modelParams["model_name"] = model_name
    trainParams["subEpochs"] = subEpochs
    trainParams["epochFr"] = epochFr
    trainParams["epochTo"] = epochTo
    trainParams["corr_indis_a"] = np.mod(epochFr, 2) if epochFr != 0 else np.mod(int(trainParams["corr_swapMode"]) + int(trainParams["corr_randMode"]), 2)
    if trainParams["applyCorr"] >= 1:
        trainParams["corrFramesAll"] = funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, featType=modelParams["dataToUse"],
                                                                    normMode=modelParams["normMode"], pcaCount=modelParams["pcaCount"], numOfSigns=numOfSigns,
                                                                    expectedFileType='Data')

    print('started training')

    directoryParams = {
        "outdir": outdir,
        "data_dir" : data_dir,
        "predictionLabelsDir": predictionLabelsDir,
        "nmi_and_acc_file_name": outdir + os.sep + exp_name + '_nmi_acc.txt'
    }

    if modelParams["trainMode"] == "rsa" or modelParams["trainMode"] == "corsa":
        funcTL.trainRNN(trainParams, modelParams, rnnParams, detailed_labels_all, model, modelTest, feat_set, labels_all, directoryParams)
    else:
        funcTL.trainFramewise(trainParams, modelParams, model, modelTest, feat_set, labels_all, directoryParams)

if __name__ == '__main__':
    main(sys.argv)