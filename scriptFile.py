import aeCluster as ae
import helperFuncs as funcH
import projRelatedHelperFuncs as prHF
import importlib as impL
import numpy as np
import os
import pandas as pd
import hmmWrapper as funcHMM
from zipfile import ZipFile
from glob import glob
import wget

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
    _confMat_preds, kluster2Classes = funcH.countPredictionsForConfusionMat(predictionsTr_nz, predClusters_nz, labelNames=None)
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

def run_dataset_paper_script(data_path_base = 'neuralNetHandImages_nos11_rs224', epochs = 30):
    error_calls = []
    for userIDTest in {2, 3, 4, 5, 6, 7}:
        for crossValidID in {1, 2, 3, 4}:  # 32
            try :
                runString = "python train_supervised.py" + \
                            " --data_path_base " + str(data_path_base) + \
                            " --epochs " + str(epochs) + \
                            " --userIDTest " + str(userIDTest) + \
                            " --crossValidID " + str(crossValidID)
                print(runString)
                os.system(runString)
            except :
                error_calls.append(runString)
    print("erroneous calls : ", error_calls)
#  run_untar()
#  run_script_combine_predictions(useNZ=True, nos=11)