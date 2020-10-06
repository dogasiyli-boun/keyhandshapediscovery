import aeCluster as ae
import helperFuncs as funcH
import projRelatedHelperFuncs as prHF
import modelFuncs as moF
import importlib as impL
import numpy as np
import os
import pandas as pd
import hmmWrapper as funcHMM
from zipfile import ZipFile
from glob import glob
import wget
import shutil
from torch import manual_seed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def study02(ep = 3):
    funcH.setPandasDisplayOpts()
    export_as_csv = False
    dcResultsFolder = funcH.getVariableByComputerName('results_dir').replace("bdResults", "dcResults")
    labelsDir = funcH.getVariableByComputerName('results_dir').replace("bdResults", "dcLabels")
    expName = 'resnet18_KMeans_pd256_clui100_11cosae-hgsk-256-11-256-st5.npy'
    labelSaveFolder = os.path.join(labelsDir, expName)
    npyFileName = 'rMF_' + expName

    fileNameFull = os.path.join(dcResultsFolder, npyFileName)
    x = np.loadtxt(fileNameFull, dtype=float, comments='#', delimiter='*', converters=None, skiprows=1, unpack=True)
    x_pd = pd.DataFrame(x.T, columns=['ep', 'tr_acc_epoch', 'nmi_lab', 'nmi_lab_nz', 'acc_lab', 'acc_lab_nz', 'nmi_pred', 'nmi_pred_nz', 'acc_pred', 'acc_pred_nz'])
    print(x_pd)
    if export_as_csv:
        fileNameFull_csv = fileNameFull.replace('.npy','.csv')
        export_csv = x_pd.to_csv (fileNameFull_csv, index = None, header=True)

    labelSaveFileName = labelSaveFolder + os.sep + 'labels_{:03d}.npz'.format(ep + 1)
    savedStuff = np.load(labelSaveFileName, allow_pickle=True)
    print('loaded', labelSaveFileName)
    print(savedStuff.files)
    labelsTrInit = savedStuff['labelsTrInit']
    predClusters = savedStuff['predClusters']
    acc_lab = savedStuff['acc_lab']
    acc_lab_nonzero = savedStuff['acc_lab_nonzero']
    predictionsTr = savedStuff['predictionsTr']
    print('ep{:03d}, acc_lab({:.5f}), acc_lab_nonzero({:.5f})'.format(ep + 1, acc_lab, acc_lab_nonzero))

    labelNames_nz = prHF.load_label_names()
    labels_true, _ = prHF.loadBaseResult("hgsk256_11_KMeans_256")
    labels_true_nz, predClusters_nz, _ = funcH.getNonZeroLabels(labels_true, predClusters)
    labels_true_nz = labels_true_nz - 1
    labelNames = labelNames_nz.copy()
    labelNames.insert(0, "None")

    _, predictionsTr_nz, _ = funcH.getNonZeroLabels(labels_true, predictionsTr)

    expIdentStr = "resnet18_cosae256_ep{:03d}".format(ep)
    expIdentStr_tr = "resnet18_cosae256_ep{:03d}_tr".format(ep)
    # klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels_true, predClusters, labelNames, expIdentStr)
    klusRet_nz, classRet_nz, _confMat_nz, c_pdf_nz, kr_pdf_nz = prHF.runForPred(labels_true_nz, predClusters_nz, labelNames_nz, expIdentStr+"_nz")
    klusRet_nz, classRet_nz, _confMat_nz, c_pdf_nz, kr_pdf_nz = prHF.runForPred(labels_true_nz, predictionsTr_nz, labelNames_nz, expIdentStr_tr+"_nz")

    #the question is what is we only used the matched samples from predictionsTr_nz==predClusters_nz
    _confMat_preds, kluster2Classes, kr_pdf, weightedPurity = funcH.countPredictionsForConfusionMat(predictionsTr_nz, predClusters_nz, labelNames=None)
    #_confMat_preds = confusion_matrix(predictionsTr_nz, predClusters_nz)
    sampleCount = np.sum(np.sum(_confMat_preds))
    acc = 100 * np.sum(np.diag(_confMat_preds)) / sampleCount
    print("acc between found correct is..", acc)

def study01(useNZ):
    numOfSigns = 11
    results_dir = funcH.getVariableByComputerName('results_dir')
    data_dir = funcH.getVariableByComputerName('data_dir')
    fileName = 'hgsk256_11_KMeans_256'
    preds = np.load(os.path.join(results_dir, 'baseResults', fileName + '.npz'))
    labels_true = np.asarray(preds['arr_0'], dtype=int)
    labels_pred = np.asarray(preds['arr_1'], dtype=int)
    labelNames = prHF.load_label_names()

    fileName_DetailedLabels = 'detailedLabels' + '_' + str(numOfSigns) + '.npy'
    detailedLabelsFileName = os.path.join(data_dir, fileName_DetailedLabels)
    detailedLabels = np.load(detailedLabelsFileName)
    print(labels_true.shape)
    print(labels_pred.shape)
    print(detailedLabels.shape)

    pred01Str = fileName + ("_nz" if useNZ else "")
    labels_true_nz, labels_pred_nz, detailedLabels_nz = funcH.getNonZeroLabels(labels_true, labels_pred, detailedLabels)
    if useNZ:
        labels_pred = labels_pred_nz
        labels_true = labels_true_nz-1
        detailedLabels = detailedLabels_nz.copy()
        detailedLabels[:, 3] = detailedLabels[:, 3]-1
    else:
        labelNames.insert(0, "None")

    funcH.getVideosToLabel(detailedLabels, labels_pred, predStr=pred01Str, labelNames=labelNames)

def runExamplePred():
    labelNames = ["cat", "fish", "hen"]
    labels_true = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) - 1
    labels_pred_1 = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 6]) - 1
    labels_pred_2 = np.array(
        [1, 1, 1, 1, 1, 1, 7, 4, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 6]) - 1
    labels_pred_3 = np.array(
        [1, 1, 1, 1, 1, 1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) - 1

    klusRet_1, classRet_1, _confMat_1, c_pdf_1, kr_pdf_1 = prHF.runForPred(labels_true, labels_pred_1, labelNames)
    klusRet_2, classRet_2, _confMat_2, c_pdf_2, kr_pdf_2 = prHF.runForPred(labels_true, labels_pred_2, labelNames)
    klusRet_3, classRet_3, _confMat_3, c_pdf_3, kr_pdf_3 = prHF.runForPred(labels_true, labels_pred_3, labelNames)

    klusRet = klusRet_1.copy().rename(columns={"value": "pred01"})
    klusRet.insert(2, "pred02", klusRet_2['value'], True)
    klusRet.insert(3, "pred03", klusRet_3['value'], True)
    print(klusRet)

    classRet = classRet_1.copy().rename(columns={"value": "pred01"})
    classRet.insert(2, "pred02", classRet_2['value'], True)
    classRet.insert(3, "pred03", classRet_3['value'], True)
    print(classRet)

    c_pdf = c_pdf_3[['class', '%f1']].sort_index().rename(columns={"class": "f1Score", "%f1": "pred01"})
    c_pdf.insert(2, "pred02", c_pdf_2[['%f1']].sort_index(), True)
    c_pdf.insert(3, "pred03", c_pdf_3[['%f1']].sort_index(), True)
    print(c_pdf)

def loadLabelsAndPreds(useNZ, nos, rs=1):
    results_dir = funcH.getVariableByComputerName('results_dir')
    labelNames = prHF.load_label_names(nos)
    cosae_hgsk_str = "cosae_pd256_wr1.0_hgsk256_" + str(nos) + "_bs16_rs" + str(rs) + "_cp2_cRM0"

    pred01Str = "hgskKmeans" + ("_nz_" if useNZ else "_") + "nos" + str(nos)
    pred02Str = "hgskCosae" + ("_nz_" if useNZ else "_") + "nos" + str(nos)
    #pred03Str = "sn256Kmeans" + ("_nz" if useNZ else "")
    #pred04Str = "sn256Cosae" + ("_nz" if useNZ else "")

    pred02_fname = os.path.join(results_dir, "results", cosae_hgsk_str, "predicted_labels004.npy")
    #pred04_fname = os.path.join(results_dir, "results", "cosae_pd256_wr1.0_sn256_11_bs16_rs1_cp2_cRM0", "predicted_labels049.npy")

    labels_true, labels_pred_1 = funcH.loadBaseResult("hgsk256_" + str(nos) + "_KMeans_256")
    labels_pred_2 = np.load(pred02_fname)
    #_, labels_pred_3 = loadBaseResult("sn256_" + str(nos) + "_KMeans_256")
    #labels_pred_4 = np.load(pred04_fname)

    print(labelNames, "\r\n", labels_true, "\r\n", labels_pred_1)

    labels_true_nz, labels_pred_nz_1, _ = funcH.getNonZeroLabels(labels_true, labels_pred_1)
    _, labels_pred_nz_2, _ = funcH.getNonZeroLabels(labels_true, labels_pred_2)
    #_, labels_pred_nz_3, _ = funcH.getNonZeroLabels(labels_true, labels_pred_3)
    #_, labels_pred_nz_4, _ = funcH.getNonZeroLabels(labels_true, labels_pred_4)

    if useNZ:
        labels_pred_1 = labels_pred_nz_1
        labels_pred_2 = labels_pred_nz_2
        #labels_pred_3 = labels_pred_nz_3
        #labels_pred_4 = labels_pred_nz_4
        labels_true = labels_true_nz-1
    else:
        labelNames.insert(0, "None")

    print(pred01Str)
    print(pred02Str)
    #print(pred03Str)
    #print(pred04Str)
    predictionsDict = []
    predictionsDict.append({"str": pred01Str, "prd": labels_pred_1})
    predictionsDict.append({"str": pred02Str, "prd": labels_pred_2})
    #predictionsDict.append({"str": pred03Str, "prd": labels_pred_3})
    #predictionsDict.append({"str": pred04Str, "prd": labels_pred_4})
    N = 2

    return labelNames, labels_true, predictionsDict, N

def run_script_combine_predictions(useNZ, nos, featUsed="hgsk256",
                     consensus_clustering_max_k = 256, verbose = False,
                     resultsToCombineDescriptorStr = "256-512-1024-9",
                     labels_preds_fold_name='/home/doga/Desktop/forBurak/wr1.0_hgsk256_11_bs16_cp2_cRM0_cSM1'):

    funcH.setPandasDisplayOpts()

    labelNames, labels, predictionsDict, cluster_runs, N = prHF.load_labels_pred_for_ensemble(useNZ=useNZ, nos=nos,
                                                                                         featUsed=featUsed,
                                                                                         labels_preds_fold_name=labels_preds_fold_name)

    prHF.ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                     consensus_clustering_max_k=consensus_clustering_max_k, useNZ=useNZ, nos=nos,
                     resultsToCombineDescriptorStr=resultsToCombineDescriptorStr,
                     labelNames=labelNames, verbose=verbose)

def run_script_hmm(n_components = 30, transStepAllow = 15, n_iter = 1000, startModel = 'firstOnly', transitionModel = 'lr'):
    numOfSigns = 11
    results_dir = funcH.getVariableByComputerName('results_dir')
    data_dir = funcH.getVariableByComputerName('data_dir')
    fileName = 'hgsk256_11_KMeans_256'
    preds = np.load(os.path.join(results_dir, 'baseResults', fileName + '.npz'))
    labels_true = np.asarray(preds['arr_0'], dtype=int)
    labels_pred = np.asarray(preds['arr_1'], dtype=int)

    fileName_DetailedLabels = 'detailedLabels' + '_' + str(numOfSigns) + '.npy'
    detailedLabelsFileName = os.path.join(data_dir, fileName_DetailedLabels)
    detailedLabels = np.load(detailedLabelsFileName)
    print(labels_true.shape)
    print(labels_pred.shape)
    print(detailedLabels.shape)

    detailed_labels_obj, summaryInfoStr = funcH.generate_detailed_labels_obj(detailedLabels)
    print(summaryInfoStr)

    verbose = 1
    _hmm_model_ = funcHMM.createHMMModel(n_components=n_components, transStepAllow=transStepAllow, n_iter=n_iter,
                                         startModel=startModel, transitionModel=transitionModel, verbose=verbose)

    maxClustID = 0
    predHMM = []
    for s in range(1, numOfSigns+1):
        frIDs_s, lengths_s, labels_s = funcH.parse_detailed_labels_obj(detailed_labels_obj, s)
        labels_pred_sign = labels_pred[frIDs_s].reshape(-1, 1)
        _hmm_model_ = funcHMM.createHMMModel(n_components=n_components, transStepAllow=transStepAllow, n_iter=n_iter,
                                             startModel=startModel, transitionModel=transitionModel, verbose=0)

        predictions_sign = funcHMM.hmm_fit_predict(_hmm_model_, labels_pred_sign, lengths=lengths_s, verbose=0)
        predictions_sign = predictions_sign + maxClustID
        print("labels_pred_sign_{}.shape{}, uniqPredLabs{}".format(s,labels_pred_sign.shape, np.unique(predictions_sign)))
        maxClustID = np.max(predictions_sign) + 1
        predHMM = np.concatenate([predHMM, predictions_sign])

    return labels_true, labels_pred, predHMM

def run_draw_confusion_script_01(figMulCnt=None, confCalcMethod = 'dnn', confusionTreshold=0.3, dataSlctID=0, numOfSigns = 11, pcaCount = 256, posterior_dim=256):
    #fs.runConfMatScript01(figMulCnt=0.60, confCalcMethod = 'count', confusionTreshold=0.2)
    #fs.runConfMatScript01(figMulCnt=0.60, confCalcMethod = 'count', confusionTreshold=0.2, dataSlctID=1, posterior_dim=256)
    results_dir = funcH.getVariableByComputerName('results_dir')
    dataToUse = 'hog'
    if dataSlctID == 0:
        predDefStr = dataToUse + str(pcaCount) + "_" + str(numOfSigns) + "_cp2_048_cosae"
        saveConfFigFileName = predDefStr + ".png"
        labels_pred = os.path.join(results_dir,
                                   'results/cosae_pd256_wr1.0_hog256_11_bs16_rs1_cp2_cRM0/predicted_labels048.npy')
    elif dataSlctID == 1:
        predDefStr = dataToUse + str(pcaCount) + "_pd" + str(posterior_dim) + "_" + str(numOfSigns) + "_base"
        saveConfFigFileName = predDefStr + ".png"
        _, labels_pred = prHF.loadBaseResult(dataToUse + str(pcaCount) + "_" + str(numOfSigns) + "_KMeans_" + str(posterior_dim))

    else:
        return
    prHF.analayzePredictionResults(labels_pred, dataToUse, pcaCount, numOfSigns,
                                   saveConfFigFileName=saveConfFigFileName, predDefStr=predDefStr,
                                   figMulCnt=figMulCnt, confCalcMethod=confCalcMethod, confusionTreshold=confusionTreshold)

def run_draw_confusion_script_02():
    _conf_mat_ = np.array([[199,  1,   0,  0],
                           [  2, 198,  0,  0],
                           [  0,  0,  97,  3],
                           [  0,  2,   2, 96]])
    class_names = ['class a', 'class b', 'class c', 'class d']
    funcH.plot_confusion_matrix(conf_mat=_conf_mat_,
                                colorbar=True,
                                show_absolute=True,
                                show_normed=True,
                                class_names=class_names,
                                saveConfFigFileName='classNames.png')

def runForPostDim(postDim=256, start_i=1, end_i=10):
    for rand_seed_x in range(start_i, end_i):
        ae.main(["aeCluster.py",
                 "--trainMode", "cosae",
                 "--dataToUse", "hgsk",
                 "--epochs", "20",
                 "--posterior_dim", str(postDim),
                 "--pcaCount", "256",
                 "--randomSeed", str(rand_seed_x),
                 "--corr_randMode", "0",
                 "--applyCorr", "2",
                 "--corr_swapMode", "1"])

# for nos in [12]: #8, 10, 11, 12
#     useNZ = True
#     runScript01(useNZ, nos, rs=10)
#     runScript01_next(useNZ, nos, rs=10)
# for nos in [10, 12]:
#     prs.run4All_createData(sign_countArr=[nos], dataToUseArr = ["hog", "skeleton", "sn"])
#     prs.createCombinedDatasets(numOfSigns=nos)
#     prs.runForBaseClusterResults(normMode='', numOfSignsArr=[nos])
#     prs.run4All_createData(sign_countArr=[nos], dataToUseArr=["hgsk"])
#     prs.runForBaseClusterResults(normMode='', numOfSignsArr=[nos], dataToUseArr=["hgsk"])

def run_untar(remove_zip=False):
    data_path_base = "neuralNetHandImages_nos11_rs224"
    filename = data_path_base + ".zip"
    url = "ftp://dogasiyli:Doga.Siyli@dogasiyli.com/hospisign.dogasiyli.com/extractedData/" + filename
    data_dir = funcH.getVariableByComputerName("base_dir")
    zip_file_name = os.path.join(data_dir,filename)
    if not os.path.isfile(zip_file_name):
        print(zip_file_name, " will be downloaded from url = ", url)
        filename = wget.download(url, out=zip_file_name)
        print("Download completed..")

    if not os.path.isdir(os.path.join(data_dir, data_path_base)):
        print(zip_file_name, " will be unzipped into = ", data_dir, " as ", data_path_base)
        with ZipFile(filename, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(data_dir)
    if remove_zip:
        print(zip_file_name, " will be deleted.")
        os.remove(filename)
        print(zip_file_name, " is deleted.")
    listOfFiles_downloaded = glob(os.path.join(data_dir, data_path_base, "imgs", "**", "*.png"))
    list_of_file_name = os.path.join(data_dir, data_path_base, "imgs", "listOfFiles.txt")
    with open(list_of_file_name) as f:
        listOfFiles_in_folder = [line.rstrip() for line in f]
    # check if they match

def run_dataset_paper_script_01(viVec=[4, 5, 6, 7], uiVec=[2], nos=11, epochs=50, modelName='resnet18'):
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    error_calls = []
    for userIDValid in viVec:
        for userIDTest in uiVec:
            try :
                runString = "python train_supervised.py" + \
                            " --modelName " + str(modelName) + \
                            " --data_path_base " + str(data_path_base) + \
                            " --epochs " + str(epochs) + \
                            " --userIDTest " + str(userIDTest) + \
                            " --userIDValid " + str(userIDValid)
                print(runString)
                os.system(runString)
            except :
                error_calls.append(runString)
    print("erroneous calls : ", error_calls)

def remove_sub_data_folds_cv(cvVec=[1, 2, 3, 4, 5], uiVec=[2, 3, 4, 5, 6, 7],
                          nos=41, random_seed=1, execute=False):
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    sup_fold = os.path.join(funcH.getVariableByComputerName("base_dir"), "sup")
    for crossValidID in cvVec:
        for userIDTest in uiVec:
            data_fold_new = "data_" + "te" + str(userIDTest) + "_cv" + str(crossValidID) + "_" + data_path_base + "_rs" + str(random_seed).zfill(2)
            data_fold_new = os.path.join(sup_fold, "data", data_fold_new)
            if os.path.isdir(data_fold_new):
                folds_2_del = funcH.getFolderList(dir2Search=data_fold_new, startString="neural")
                print("+ui({:d}),cv({:d})".format(userIDTest, crossValidID))
                for f in folds_2_del:
                    print("--delete fold = <", os.path.join(data_fold_new, f), ">")
                    if execute:
                        shutil.rmtree(os.path.join(data_fold_new, f))

def remove_sub_data_folds_va(viVec=[2, 3, 4, 5, 6 ,7], uiVec=[2, 3, 4, 5, 6, 7],
                             nos=41, random_seed=1, execute=False):
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    sup_fold = os.path.join(funcH.getVariableByComputerName("base_dir"), "sup")
    for userIDValid in viVec:
        for userIDTest in uiVec:
            data_fold_new = "data_" + "te" + str(userIDTest) + "_va" + str(userIDValid) + "_" + data_path_base + "_rs" + str(random_seed).zfill(2)
            data_fold_new = os.path.join(sup_fold, "data", data_fold_new)
            if os.path.isdir(data_fold_new):
                folds_2_del = funcH.getFolderList(dir2Search=data_fold_new, startString="neural")
                print("+ui({:d}),va({:d})".format(userIDTest, userIDValid))
                for f in folds_2_del:
                    print("--delete fold = <", os.path.join(data_fold_new, f), ">")
                    if execute:
                        shutil.rmtree(os.path.join(data_fold_new, f))


def run_dataset_paper_script(viVec=[4, 5, 6, 7], uiVec=[2], nos=11, epochs=50, modelName='resnet18',
                             rsdf=[False, True], random_seed=1):
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    error_calls = []
    for userIDValid in viVec:
        for userIDTest in uiVec:
            if rsdf[0]:
                remove_sub_data_folds_va(viVec=[userIDValid], uiVec=[userIDTest], nos=nos,
                                         random_seed=random_seed, execute=True)
            try:
                runString = "python train_supervised.py" + \
                            " --modelName " + str(modelName) + \
                            " --data_path_base " + str(data_path_base) + \
                            " --epochs " + str(epochs) + \
                            " --userIDTest " + str(userIDTest) + \
                            " --userIDValid " + str(userIDValid)
                print(runString)
                os.system(runString)
            except:
                error_calls.append(runString)

            if rsdf[1]:
                remove_sub_data_folds_va(viVec=[userIDValid], uiVec=[userIDTest], nos=nos,
                                         random_seed=random_seed, execute=True)

    print("erroneous calls : ", error_calls)
#  run_untar()
#  run_script_combine_predictions(useNZ=True, nos=11)

def change_fold_names(cvVec=[1, 2, 3, 4, 5], uiVec=[2, 3, 4, 5, 6, 7],
                      nos=41, epochs=50, random_seed=1, modelName='resnet18',
                      delete_sub_data_folds=False, execute=False):
    data_path_base = "neuralNetHandImages_nos" + str(nos) + "_rs224"
    sup_fold = os.path.join(funcH.getVariableByComputerName("base_dir"), "sup")
    for crossValidID in cvVec:
        for userIDTest in uiVec:
            pred_fold_old = "pred_" + "te" + str(userIDTest) + "_cv" + str(crossValidID) + "_" + modelName + data_path_base
            data_fold_old = "data_" + "te" + str(userIDTest) + "_cv" + str(crossValidID) + "_" + modelName + data_path_base
            # check if there are 61 items under pred_fold_old
            pred_fold_abs_old = os.path.join(sup_fold, pred_fold_old)

            pred_fold_new = "pred_te" + str(userIDTest) + "_cv" + str(crossValidID) + "_" + modelName + "_" + data_path_base + "_rs" + str(random_seed).zfill(2)
            data_fold_new = str(data_fold_old + "_rs" + str(random_seed).zfill(2)).replace(modelName, "")

            if os.path.isdir(pred_fold_abs_old):
                fl = funcH.getFileList(dir2Search=pred_fold_abs_old, startString="ep", endString=".npy")
                if len(fl) <= epochs+1:
                    #change pred folder name
                    pred_fold_abs_to = os.path.join(sup_fold, "pred", pred_fold_new)
                    print("move -> ", pred_fold_abs_old, "-to-", pred_fold_abs_to)
                    if execute:
                        shutil.move(pred_fold_abs_old, pred_fold_abs_to)
                        print("++moved -> ", pred_fold_abs_old, "-to-", pred_fold_abs_to)

                    data_fold_abs_old = os.path.join(sup_fold, data_fold_old)
                    data_fold_abs_new = os.path.join(sup_fold, "data", data_fold_new)
                    if delete_sub_data_folds:
                        folds_2_del = funcH.getFolderList(dir2Search=data_fold_abs_old, startString="neural")
                        for f in folds_2_del:
                            print("delete fold = <", os.path.join(data_fold_abs_old, f), ">")
                            if execute:
                                shutil.rmtree(os.path.join(data_fold_abs_old, f))
                                print("--deleted fold = <", os.path.join(data_fold_abs_old, f), ">")
                    print("move--> ", data_fold_abs_old, "-to-", data_fold_abs_new)
                    if execute:
                        shutil.move(data_fold_abs_old, data_fold_abs_new)
                        print("++moved--> ", data_fold_abs_old, "-to-", data_fold_abs_new)
                else:
                    print(pred_fold_old, "- active")
            else:
                print(pred_fold_old, "- ?")


def mlp_study_01(dataIdent="hgsk", pca_dim=256, nos=11, validUser=3, epochCnt=10, testUser=2, verbose=2, model_= None):
    ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(dataIdent=dataIdent, pca_dim=pca_dim, nos=nos, verbose=verbose)
    dl_tr, dl_va, dl_te = prHF.prepare_data_4(ft, lb, lb_sui, validUser=validUser, testUser=testUser, useNZ=False)
    hidCounts = [128, 64]
    np.random.seed(0)
    uniqLabs = np.unique(lb)
    classCount = len(uniqLabs)
    print("uniqLabs=", uniqLabs, ", classCount_="+dataIdent, classCount)

    if model_ is None:
        model_ = moF.MLP(ft.shape[1], hidCounts, classCount)

    accvectr, accvecva, accvecte = model_.train_evaluate_trvate(dl_tr, dl_va, dl_te, epochCnt=epochCnt)

    bestVaID = np.argmax(accvecva)
    bestTeID = np.argmax(accvecte)
    formatStr = "5.3f"
    print(("bestVaID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID,accvecva[bestVaID],accvecte[bestVaID]))
    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID,accvecva[bestTeID],accvecte[bestTeID]))
    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1], accvecte[-1]))

def mlp_study_02(dataIdent="hgsk", pca_dim=256, nos=11, validUser=3, epochCnt=10, testUser=2, verbose=2, model_= None):
    ft, lb, lb_sui = prHF.combine_pca_hospisign_data(dataIdent=dataIdent, pca_dim=pca_dim, nos=nos, verbose=verbose)
    dl_tr, dl_va, dl_te = prHF.prepare_data_4(ft, lb, lb_sui, validUser=validUser, testUser=testUser, useNZ=False)

    hidStateDict_01 = {"dimOut": 128, "initMode": "kaiming_uniform_", "act": "relu"}
    hidStateDict_02 = {"dimOut": 64, "initMode": "kaiming_uniform_", "act": "relu"}
    hidStatesDict = {
        "01": hidStateDict_01,
        "02": hidStateDict_02,
    }

    np.random.seed(0)
    uniqLabs = np.unique(lb)
    classCount = len(uniqLabs)
    print("uniqLabs=", uniqLabs, ", classCount_="+dataIdent, classCount)

    if model_ is None:
        model_ = moF.MLP_Dict(ft.shape[1], hidStatesDict, classCount)

    accvectr, accvecva, accvecte = model_.train_evaluate_trvate(dl_tr, dl_va, dl_te, epochCnt=epochCnt)

    bestVaID = np.argmax(accvecva)
    bestTeID = np.argmax(accvecte)
    formatStr = "5.3f"
    print(("bestVaID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID,accvecva[bestVaID],accvecte[bestVaID]))
    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID,accvecva[bestTeID],accvecte[bestTeID]))
    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1], accvecte[-1]))

def get_hid_state_vec(hidStateID):
    hid_state_cnt_vec = [2048, 1024, 1024, 512, 512, 256, 256]
    if hidStateID == 1:
        hid_state_cnt_vec = [256, 256]
    elif hidStateID == 2:
        hid_state_cnt_vec = [512, 512]
    elif hidStateID == 3:
        hid_state_cnt_vec = [512, 512, 256, 256]
    elif hidStateID == 4:
        hid_state_cnt_vec = [1024, 512, 512, 256]
    elif hidStateID == 5:
        hid_state_cnt_vec = [64, 64, 64, 64]
    elif hidStateID == 6:
        hid_state_cnt_vec = [128, 128, 128, 128]
    elif hidStateID == 7:
        hid_state_cnt_vec = [256, 256, 256, 256]
    elif hidStateID == 8:
        hid_state_cnt_vec = [512, 512, 512, 512]
    elif hidStateID == 9:
        hid_state_cnt_vec = [128, 128]

    return hid_state_cnt_vec

def mlp_study_03(dropout_value, hidStateID, nos, dataIdent_vec, pca_dim = 256, verbose = 0, validationUserVec = [2, 3, 4, 5, 6, 7], testUserVec = [2, 3, 4, 5, 6, 7], resultFolder=os.path.join(funcH.getVariableByComputerName('desktop_dir'),'resultSome'), rs_range=1):
    impL.reload(moF)

    for dataIdent in dataIdent_vec:
        # prepare and get the data along with labels and necessary variables
        ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(dataIdent=dataIdent, pca_dim=pca_dim, nos=nos, verbose=verbose)

        doStr = ""
        if dropout_value is not None:
            doStr = "_do{:4.2f}".format(dropout_value)

        hid_state_cnt_vec = get_hid_state_vec(hidStateID)
        hidStatesDict = moF.create_hidstate_dict(hid_state_cnt_vec, init_mode_vec=None, act_vec=None)

        for userVa in validationUserVec:
            for userTe in testUserVec:
                if userVa == userTe:
                    continue
                print("*****\nprepare_data_4 : \nteUser=", userTe, ", vaUser=", userVa)
                dl_tr, dl_va, dl_te = prHF.prepare_data_4(ft, lb, lb_sui, validUser=userVa, testUser=userTe)
                for rs in range(rs_range):
                    result_file_name = 'di' + dataIdent + '_nos' + str(nos) + '_te' + str(userTe) + '_va' + str(userVa) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + '.npz'
                    result_file_name_full = os.path.join(resultFolder, result_file_name)
                    result_file_exist = os.path.exists(result_file_name_full)
                    best_model_file_name = result_file_name.replace('.npz', '.model')
                    best_model_file_name_full = result_file_name_full.replace('.npz', '.model')
                    model_file_exist = os.path.exists(best_model_file_name_full)
                    if result_file_exist:
                        print("RESULT FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", result_file_name)
                    else:
                        print("RESULT FILE DOESNT EXIST +++")
                    if model_file_exist:
                        print("MODEL FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", best_model_file_name)
                    else:
                        print("MODEL FILE DOESNT EXIST +++")

                    try:
                        if result_file_exist and model_file_exist:
                            print("**********teUser=", userTe, ", vaUser=", userVa, " -- skipping : ", result_file_name, "-", best_model_file_name)
                            class_names = np.asarray(lb_map["khsName"])
                            df_slctd_table = prHF.get_result_table_out(result_file_name_full, class_names)
                            continue
                    except:
                        pass

                    print("***************\nteUser=", userTe, ", vaUser=", userVa, ", rs=", rs)

                    # check if exist. if yes go on

                    print("teUser=", userTe, ", vaUser=", userVa, ", rs=", rs)
                    np.random.seed(rs)
                    manual_seed(rs)
                    uniqLabs = np.unique(lb)
                    classCount = len(uniqLabs)
                    print("uniqLabs=", uniqLabs, ", classCount_=" + dataIdent, classCount)

                    model_ = moF.MLP_Dict(ft.shape[1], hidStatesDict, classCount, dropout_value=dropout_value)

                    accvectr, accvecva, accvecte, preds_best, labels_best = model_.train_evaluate_trvate(dl_tr, dl_va, dl_te, epochCnt=30, saveBestModelName=best_model_file_name_full)

                    print("dataIdent=", dataIdent)
                    print("nos=", nos)
                    print("userVa=", userVa)
                    print("rs=", rs)
                    print("hid_state_cnt_vec=", hid_state_cnt_vec)

                    bestVaID = np.argmax(accvecva)
                    bestTeID = np.argmax(accvecte)
                    formatStr = "5.3f"
                    print(("bestVaID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestVaID, accvecva[bestVaID], accvecte[bestVaID]))
                    print(("bestTeID({:" + formatStr + "}),vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(bestTeID, accvecva[bestTeID], accvecte[bestTeID]))
                    print(("last, vaAcc({:" + formatStr + "}),teAcc({:" + formatStr + "})").format(accvecva[-1],accvecte[-1]))

                    np.savez(result_file_name_full, dataIdent_=dataIdent, testUser_=userTe, validUser_=userVa,
                             hid_state_cnt_vec_=hid_state_cnt_vec, accvectr_=accvectr, accvecva_=accvecva,
                             accvecte_=accvecte, preds_best_=preds_best, labels_best_=labels_best, allow_pickle=True)
                    asasas = np.load(result_file_name_full, allow_pickle=True)
                    print(asasas.files, "\n***************")
                print("*****\n")

def mlp_analyze_result_hgsnsk(userTe, userVa, dropout_value, dataIdent, nos, rs=0, hidStateID = 0, pca_dim = 256, verbose = 0, resultFolder = os.path.join(funcH.getVariableByComputerName('desktop_dir'),'resultSome')):
    impL.reload(moF)
    impL.reload(prHF)
    ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(dataIdent=dataIdent, pca_dim=pca_dim, nos=nos,
                                                             verbose=verbose)
    class_names = np.asarray(lb_map["khsName"])
    classCount = len(class_names)

    doStr = ""
    if dropout_value is not None:
        doStr = "_do{:4.2f}".format(dropout_value)

    result_file_name = 'di' + dataIdent + '_nos' + str(nos) + '_te' + str(userTe) + '_va' + str(userVa) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + '.npz'
    result_file_name_full = os.path.join(resultFolder, result_file_name)
    result_file_exist = os.path.exists(result_file_name_full)
    best_model_file_name_full = result_file_name_full.replace('.npz', '.model')
    model_file_exist = os.path.exists(best_model_file_name_full)
    if result_file_exist:
        print("RESULT FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", result_file_name)
    else:
        print("RESULT FILE DOESNT EXIST +++", result_file_name)
    if model_file_exist:
        print("MODEL FILE EXIST +++ teUser=", userTe, ", vaUser=", userVa, ": ", best_model_file_name_full)
    else:
        print("MODEL FILE DOESNT EXIST +++", best_model_file_name_full)

    df_slctd_table = []
    if result_file_exist:
        df_slctd_table = prHF.get_result_table_out(result_file_name_full, class_names)

    model_exports = []
    if model_file_exist:
        hid_state_cnt_vec = get_hid_state_vec(hidStateID)
        hidStatesDict = moF.create_hidstate_dict(hid_state_cnt_vec, init_mode_vec=None, act_vec=None)
        model_ = moF.MLP_Dict(ft.shape[1], hidStatesDict, classCount)
        print("loading best model--", best_model_file_name_full)
        model_.load_model(best_model_file_name_full)

        dl_tr, dl_va, dl_te = prHF.prepare_data_4(ft, lb, lb_sui, validUser=userVa, testUser=userTe)
        print("exporting train features...")
        acc_tr, preds_tr, labs_tr, final_layer_tr = model_.export_final_layer(dl_tr)
        print("exporting validation features...")
        acc_va, preds_va, labs_va, final_layer_va = model_.export_final_layer(dl_va)
        print("exporting test features...")
        acc_te, preds_te, labs_te, final_layer_te = model_.export_final_layer(dl_te)
        model_exports = {
            "tr": {"acc": acc_tr, "preds": preds_tr, "labs": labs_tr, "final_layer": final_layer_tr},
            "va": {"acc": acc_va, "preds": preds_va, "labs": labs_va, "final_layer": final_layer_va},
            "te": {"acc": acc_te, "preds": preds_te, "labs": labs_te, "final_layer": final_layer_te},
        }
    return df_slctd_table, model_exports

def get_from_model(model_exports_x, model_str, normalizationMode, data_va_te_str, verbose=0):
    ft = model_exports_x[data_va_te_str]["final_layer"]
    pr = model_exports_x[data_va_te_str]["preds"]
    la = model_exports_x[data_va_te_str]["labs"]
    if verbose > 0:
        print(ft.shape)
        print(pr.shape)
        print(la.shape)
        print(model_str, ", ", data_va_te_str, " acc = ", "{:5.3f}".format(accuracy_score(la, pr)), ", normMode=", normalizationMode)
    if normalizationMode == 'max':
        ft_n = funcH.normalize2(ft, normMode='nm', axis=1)
    elif normalizationMode == 'sum':
        ft_n = funcH.normalize2(ft, normMode='ns', axis=1)
    elif normalizationMode == 'softmax':
        ft_n = funcH.softmax(ft.T).T
    else:
        ft_n = ft
    return ft_n, pr, la

def test_normalization(n, c, axis, rs=0):
    print("n(", n, ") samples and c(", c, ") classes")
    np.random.seed(rs)
    X = np.random.uniform(0, 1, [n, c])
    print("X =\n", X)

    Xmn  = funcH.normalize2(X, normMode='nm', axis=axis)
    Xsum = funcH.normalize2(X, normMode='ns', axis=axis)
    Xsof = funcH.normalize2(X, normMode='softmax', axis=axis)

    pred_max = np.argmax(Xmn, axis=1)
    pred_sum = np.argmax(Xsum, axis=1)
    pred_sof = np.argmax(Xsof, axis=1)
    print("predictions_max=", pred_max)
    print("predictions_sum=", pred_sum)
    print("predictions_som=", pred_sof)

    return X, Xmn, Xsum, Xsof

def check_model_exports(model_exports_x, model_export_string):
    print("model_exports_", model_export_string)
    print(model_exports_x["tr"]["acc"], model_exports_x["va"]["acc"], model_exports_x["te"]["acc"])
    print("tr-", model_exports_x["tr"]["final_layer"].shape)
    print("va-", model_exports_x["va"]["final_layer"].shape)
    print("te-", model_exports_x["te"]["final_layer"].shape)

def mlp_study_score_fuse(userTe, userVa, dropout_value, rs, hidStateID, nos, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"]):
    model_export_dict = {}
    for data_ident in data_ident_vec:
        df_slctd, model_exports = mlp_analyze_result_hgsnsk(userTe=userTe, userVa=userVa, dropout_value=dropout_value, dataIdent=data_ident, nos=nos, rs=rs, hidStateID=hidStateID)
        dict_cur = {"df_slctd_table": df_slctd, "model_export": model_exports}
        model_export_dict[data_ident] = dict_cur
    for data_ident in data_ident_vec:
        check_model_exports(model_export_dict[data_ident]["model_export"], data_ident)
    return model_export_dict

def mlp_study_score_fuse_apply(model_export_dict, defStr, data_va_te_str, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"]):
    ft_comb_ave = None
    ft_comb_max = None
    ft_comb_sof = None

    df_final = pd.DataFrame({"khsName": model_export_dict["hog"]["df_slctd_table"]["khsName"].sort_index()})
    acc_vec_all = {}
    for data_ident in data_ident_vec:
        #print("****\n", defStr, "\n", data_ident)
        ft_max, preds_te, labels_xx = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode="max", data_va_te_str=data_va_te_str)
        ft_sof, _, _ = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode="softmax", data_va_te_str=data_va_te_str)
        ft_none, _, _ = get_from_model(model_export_dict[data_ident]["model_export"], model_str=data_ident, normalizationMode=None, data_va_te_str=data_va_te_str)

        print(data_ident+data_va_te_str+"_acc = ", "{:5.3f}".format(accuracy_score(labels_xx, preds_te)))

        df_final = pd.concat([df_final, model_export_dict[data_ident]["df_slctd_table"]["F1_Score"].sort_index()], axis = 1).rename(columns={"F1_Score": data_ident})
        ft_comb_ave = ft_max if ft_comb_ave is None else ft_max+ft_comb_ave
        ft_comb_max = ft_none if ft_comb_max is None else np.maximum(ft_none, ft_comb_max)
        ft_comb_sof = ft_sof if ft_comb_sof is None else ft_sof+ft_comb_sof

        preds_max = np.argmax(ft_max, axis=1)
        preds_sof = np.argmax(ft_sof, axis=1)

        acc_max = accuracy_score(labels_xx, preds_max)
        acc_sof = accuracy_score(labels_xx, preds_sof)
        if acc_max-acc_sof != 0.0:
            print(defStr, data_ident, ", max norm acc = ", "{:5.3f}".format(acc_max))
            print(defStr, data_ident, ", softmax norm acc = ", "{:5.3f}".format(acc_sof))
        #print("****")
        acc_vec_all[data_ident] = acc_max

    classNames = np.asarray(model_export_dict["hog"]["df_slctd_table"]["khsName"].sort_index())

    pr_comb_ave = np.argmax(ft_comb_ave, axis=1)
    pr_comb_max = np.argmax(ft_comb_max, axis=1)
    pr_comb_sof = np.argmax(ft_comb_sof, axis=1)
    acc_ave = accuracy_score(labels_xx, pr_comb_ave)
    acc_max = accuracy_score(labels_xx, pr_comb_max)
    acc_sof = accuracy_score(labels_xx, pr_comb_sof)
    print(defStr, data_va_te_str, "comb-(AVE) acc = {:6.4f}".format(acc_ave))
    print(defStr, data_va_te_str, "comb-(MAX) acc = {:6.4f}".format(acc_max))
    print(defStr, data_va_te_str, "comb-(SOFTMAX) acc = {:6.4f}".format(acc_sof))

    str_j = '_'.join(data_ident_vec)
    acc_vec_all[str_j+'_ave'] = acc_ave
    acc_vec_all[str_j+'_max'] = acc_max
    acc_vec_all[str_j+'_sof'] = acc_sof

    conf_mat_ave = confusion_matrix(labels_xx, pr_comb_ave)
    conf_mat_max = confusion_matrix(labels_xx, pr_comb_max)
    conf_mat_sof = confusion_matrix(labels_xx, pr_comb_sof)

    while len(classNames) < conf_mat_max.shape[0] or len(classNames) < conf_mat_sof.shape[0]:
        classNames = np.hstack((classNames, "wtf"))

    cmStats_ave, df_ave = funcH.calcConfusionStatistics(conf_mat_ave, categoryNames=classNames, selectedCategories=None, verbose=0)
    cmStats_max, df_max = funcH.calcConfusionStatistics(conf_mat_max, categoryNames=classNames, selectedCategories=None, verbose=0)
    cmStats_sofx, df_sof = funcH.calcConfusionStatistics(conf_mat_sof, categoryNames=classNames, selectedCategories=None, verbose=0)
    df_final = pd.concat([df_final, df_ave["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_ave"})
    df_final = pd.concat([df_final, df_max["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_max"})
    df_final = pd.concat([df_final, df_sof["F1_Score"].sort_index()], axis=1).rename(columns={"F1_Score": "df_sof"})

    df_final.to_csv(os.path.join(funcH.getVariableByComputerName('desktop_dir'), "comb", "comb_" + defStr + data_va_te_str + "_all.csv"))

    print(df_final)
    print(acc_vec_all)

    results_dict = {
        "conf_mat_max": conf_mat_max,
        "conf_mat_sof": conf_mat_sof,
        "df_final": df_final,
        "df_ave": df_ave,
        "df_max": df_max,
        "df_sof": df_sof,
        "acc_vec_all": acc_vec_all,
    }
    return results_dict

def save_df_tables_for_te_va(userTe, userVa, dropout_value, hidStateID, nos, model_export_dict_folder=os.path.join(funcH.getVariableByComputerName('desktop_dir'),'modelExportDicts')):
    defStr = "te" + str(userTe) + "_va" + str(userVa) + "_nos" + str(nos)
    model_export_dict_fname = os.path.join(model_export_dict_folder, defStr + "_mex.npy")
    if os.path.exists(model_export_dict_fname):
        print("loading model_export_dict from :", model_export_dict_fname)
        model_export_dict = np.load(model_export_dict_fname, allow_pickle=True).item()
        print("loaded model_export_dict from :", model_export_dict_fname)
    else:
        print("creating model_export_dict for :", defStr)
        model_export_dict = mlp_study_score_fuse(userTe=userTe, userVa=userVa, dropout_value=dropout_value, rs=0, hidStateID=hidStateID, nos=nos, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk","hgsnsk"])
        np.save(model_export_dict_fname, model_export_dict, allow_pickle=True)
        print("saving model_export_dict at :", model_export_dict_fname)
    return model_export_dict, defStr

def append_to_all_results(results_dict, index_name, dropout_value, rs, hidStateID, nos, data_va_te_str,
                          model_export_dict_folder=os.path.join(funcH.getVariableByComputerName('desktop_dir'),'modelExportDicts')):
    columns = ["hog", "sk", "sn", "hgsk", "hgsn", "snsk", "hgsnsk", "hogsk", "hogskhgsk", "hogsn", "hogsnsk", "ALLax", "ALLmx", "ALLsm"]
    doStr = ""
    if dropout_value is not None:
        doStr = "_do{:4.2f}".format(dropout_value)
    all_results_filename = 'nos' + str(nos) + '_rs' + str(rs) + '_hs' + str(hidStateID) + doStr + data_va_te_str + '.npy'
    all_results_filename = os.path.join(model_export_dict_folder, all_results_filename)

    if os.path.exists(all_results_filename):
        print("loading...", all_results_filename)
        all_results = pd.read_pickle(all_results_filename)
        print(all_results_filename, " loaded : \n", all_results)
    else:
        all_results = pd.DataFrame(index=None, columns=columns)
        all_results.to_pickle(all_results_filename)
        print(all_results_filename, " saved as empty list")

    if index_name not in all_results.index:
        a = pd.DataFrame(np.nan, index=[index_name], columns=columns)
        all_results = all_results.append(a)
        all_results.to_pickle(all_results_filename)
        print(all_results_filename, " saved adding index_name = ", index_name)

    keys_added = list()
    for key, value in results_dict["acc_vec_all"].items():
        # keyN = keyN.replace('hog','hg')
        keyN = key.replace('_', '')
        keyN = keyN.replace('hogsnskhgskhgsnsnskhgsnsk', 'ALL')
        keyN = keyN.replace('ALLave', 'ALLax')
        keyN = keyN.replace('ALLmax', 'ALLmx')
        keyN = keyN.replace('ALLsof', 'ALLsm')
        keyN = keyN.replace('max', '')
        keyN = keyN.replace('ave', '')
        try:
            if all_results[keyN][index_name] != value:
                all_results[keyN][index_name] = value
                print("added", keyN, value)
                all_results.loc[index_name, keyN] = value
                print(all_results[keyN][index_name])
                keys_added.append(keyN)
            else:
                print("(same)skipped", key, keyN, value)
        except:
            print("(error)skipped", key, keyN, value)
    if len(keys_added) > 0:
        print(all_results_filename, " updated by index_name = ", index_name, ", keys_added=", keys_added)
        all_results.to_pickle(all_results_filename)
        print("updated : \n", all_results)
    return all_results

def append_to_all_results_dv_loop(userTe, userVa, nos, dv, data_va_te_str, dropout_value, rs, hidStateID):
    model_export_dict, defStr = save_df_tables_for_te_va(userTe=userTe, userVa=userVa, dropout_value=dropout_value, hidStateID=hidStateID, nos=nos)
    index_name = "te" + str(userTe) + "_va" + str(userVa) + "_nos" + str(nos)
    for data_ident_vec in dv:
        str_j = '_'.join(data_ident_vec)
        defStr2 = defStr + "_" + str_j
        print("defStr=", defStr)
        results_dict = mlp_study_score_fuse_apply(model_export_dict, defStr2, data_va_te_str=data_va_te_str, data_ident_vec=data_ident_vec)
        all_results = append_to_all_results(results_dict, index_name, dropout_value=dropout_value, rs=rs,
                                            hidStateID=hidStateID, nos=nos, data_va_te_str=data_va_te_str)
    return all_results

def cluster_resluts_journal01(nos = 11):
    labelNames = prHF.load_label_names(nos)
    results_dir = funcH.getVariableByComputerName('results_dir')  # '/media/dg/SSD_Data/DataPath/bdResults'
    baseLineResultFolder = os.path.join(results_dir, 'baseResults')  # '/media/dg/SSD_Data/DataPath/bdResults/baseResults'
    baseResFiles = funcH.getFileList(baseLineResultFolder, startString="", endString=".npz", sortList=False)
    for f in baseResFiles:
        if not str(f).__contains__("_" + str(nos)):
            continue
        if not str(f).__contains__("Kmeans"):
            continue
        labels_true, labels_pred = prHF.loadBaseResult(f)
        labels_true, labels_pred, _ = funcH.getNonZeroLabels(labels_true, labels_pred)
        labels_true = labels_true - 1
        _confMat_preds, kluster2Classes, kr_pdf, weightedPurity = funcH.countPredictionsForConfusionMat(labels_true, labels_pred, labelNames=labelNames)
        sampleCount = np.sum(np.sum(_confMat_preds))
        acc = 100 * np.sum(np.diag(_confMat_preds)) / sampleCount
        meanPurity = np.mean(np.asarray(kr_pdf["%purity"]))

        f = f.replace("_" + str(nos),"")
        f = f.replace("_Kmeans","")
        print("f({:s}),n({:d},{:d}), acc({:5.3f}), meanPurity({:5.3f}), weightedPurity({:5.3f})".format(f, labels_true.shape[0], labels_pred.shape[0], acc, meanPurity, weightedPurity))
        print("****")

def cluster_journal01_aug2020(rs, nos, data_ident_vec=["hog", "sn", "sk", "hgsk", "hgsn", "snsk", "hgsnsk"], clustCntVec=[128, 256], verbose=0, pca_dim=256):
    for data_ident in data_ident_vec:
        pca_dim_2_use = pca_dim
        if (data_ident == 'skeleton' or data_ident == 'sk') and pca_dim > 112:
            pca_dim_2_use = 96
        ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(dataIdent=data_ident, nos=nos, pca_dim=pca_dim_2_use, verbose=verbose)
        resultDict = prHF.runClusteringOnFeatSet_Aug2020(ft, lb, lb_map, dataToUse=data_ident, numOfSigns=nos,
                                                         pcaCount=pca_dim_2_use, clustCntVec=clustCntVec, randomSeed=rs)
    prHF.traverseBaseResultsFolder_Aug2020()
    return

def combine_clusters_Aug2020(nos = 11, dataToUseVec = ["hog", "sn", "sk"], clustCntVec=[256], consensus_clustering_max_k=256):
    impL.reload(prHF)
    ft, lb, lb_sui, lb_map = prHF.combine_pca_hospisign_data(dataIdent="sk", nos=nos, pca_dim=None, verbose=0)
    del(ft, lb, lb_sui)
    class_names = np.asarray(lb_map["khsName"])
    del (lb_map)

    resultsToCombineDescriptorStr = '|'.join(dataToUseVec)
    labelNames, labels, predictionsDict, cluster_runs, N = prHF.load_labels_pred_for_ensemble_Aug2020(
                    class_names, nos=nos, clustCntVec=clustCntVec, dataToUseVec=dataToUseVec)
    prHF.ensemble_cluster_analysis(cluster_runs, predictionsDict, labels,
                                   consensus_clustering_max_k=consensus_clustering_max_k, useNZ=False, nos=nos,
                                   resultsToCombineDescriptorStr=resultsToCombineDescriptorStr,
                                   labelNames=labelNames, verbose=True)