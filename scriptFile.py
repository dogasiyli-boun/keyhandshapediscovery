import dataLoaderFuncs as funcD
import projRelatedScripts as prs
from sklearn.metrics import confusion_matrix

import helperFuncs as funcH
import projRelatedHelperFuncs as prHF
import numpy as np
import os
import pandas as pd
import Cluster_Ensembles as CE  # sudo apt-get install metis
import hmmWrapper as funcHMM
import ensembleFuncs as funcEnsemble
import datetime
import time

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

def runScript01(useNZ, nos, rs=1):
    funcH.setPandasDisplayOpts()

    labelNames, labels_true, predictionsDict, N = loadLabelsAndPreds(useNZ, nos, rs=rs)
    print(predictionsDict[0]["str"])
    print(predictionsDict[1]["prd"])

    cluster_runs = None
    for i in range(0, N):
        cluster_runs = funcH.append_to_vstack(cluster_runs, predictionsDict[i]["prd"], dtype=int)

    consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=False, N_clusters_max=256)

    predCombined = "combined" + ("_nz" if useNZ else "") + "_" + str(nos)
    predictionsDict.append({"str": predCombined, "prd": consensus_clustering_labels})
    cluster_runs = funcH.append_to_vstack(cluster_runs, consensus_clustering_labels, dtype=int)

    resultsDict = []
    for i in range(0, N+1):
        klusRet, classRet, _confMat, c_pdf, kr_pdf = prHF.runForPred(labels_true, predictionsDict[i]["prd"], labelNames, predictionsDict[i]["str"])
        resultsDict.append({"klusRet": klusRet, "classRet": classRet,
                            "_confMat": _confMat, "c_pdf": c_pdf, "kr_pdf": kr_pdf})

    klusRet = resultsDict[0]["klusRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N+1):
        klusRet.insert(i+1, predictionsDict[i]["str"], resultsDict[i]["klusRet"]['value'], True)
    print("\r\ncluster metrics comparison\r\n")
    print(klusRet)
    print("\r\n")

    classRet = resultsDict[0]["classRet"].copy().rename(columns={"value": predictionsDict[0]["str"]})
    for i in range(1, N+1):
        classRet.insert(i+1, predictionsDict[i]["str"], resultsDict[i]["classRet"]['value'], True)
    print("\r\nclassification metrics comparison\r\n")
    print(classRet)
    print("\r\n")

    c_pdf = resultsDict[0]["c_pdf"][['class', '%f1']].sort_index().rename(columns={"class": "f1Score", "%f1": predictionsDict[0]["str"]})
    for i in range(1, N+1):
        c_pdf.insert(i+1, predictionsDict[i]["str"], resultsDict[i]["c_pdf"][['%f1']].sort_index(), True)
    print("\r\nf1 score comparisons for classes\r\n")
    print(c_pdf)
    print("\r\n")

    print('calc_ensemble_driven_cluster_index - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    eci_vec, clusterCounts = funcEnsemble.calc_ensemble_driven_cluster_index(cluster_runs=cluster_runs)
    elapsed = time.time() - t
    print('calc_ensemble_driven_cluster_index - elapsedTime(', elapsed, '), ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_LWCA_matrix - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    lwca_mat = funcEnsemble.create_LWCA_matrix(cluster_runs, eci_vec=eci_vec, verbose=0)
    elapsed = time.time() - t
    print('create_LWCA_matrix - elapsedTime(', elapsed, '), ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print('create_quality_vec - started at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = time.time()
    quality_vec = funcEnsemble.calc_quality_weight_basic_clustering(cluster_runs, logType=0, verbose=0)
    elapsed = time.time() - t
    print('create_quality_vec - elapsedTime(', elapsed, '), ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    results_dir = funcH.getVariableByComputerName("results_dir")
    predictResultFold = os.path.join(results_dir, "predictionResults")
    resultsToCombine_FileName = "klusterResults_" + str(nos) + ".npz"
    resultsToCombine_FileName = os.path.join(predictResultFold, resultsToCombine_FileName)
    np.savez(resultsToCombine_FileName, lwca_mat=lwca_mat, predictionsDict=predictionsDict, resultsDict=resultsDict, eci_vec=eci_vec, clusterCounts=clusterCounts, quality_vec=quality_vec)

def runScript01_next(useNZ, nos, rs=1):
    results_dir = funcH.getVariableByComputerName("results_dir")
    predictResultFold = os.path.join(results_dir, "predictionResults")
    resultsToCombine_FileName = "klusterResults_" + str(nos) + ".npz"
    resultsToCombine_FileName = os.path.join(predictResultFold, resultsToCombine_FileName)
    loadedResults = np.load(resultsToCombine_FileName, allow_pickle=True)

    lwca_mat = loadedResults["lwca_mat"]
    predictionsDict = loadedResults["predictionsDict"]
    resultsDict = loadedResults["resultsDict"]
    eci_vec = loadedResults["eci_vec"]
    clusterCounts = loadedResults["clusterCounts"]
    quality_vec = loadedResults["quality_vec"]

    labelNames, labels_true, _, _ = loadLabelsAndPreds(useNZ, nos, rs=rs)
    N = resultsDict.shape[0]
    sampleCntToPick = np.array([1, 3, 5, 10], dtype=int)
    columns = ['1', '3', '5', '10']
    colCnt = 4

    #cluster_runs_cmbn = []
    for i in range(0, N):
        kr_pdf_cur = resultsDict[i]["kr_pdf"]
        eci_vec_cur = eci_vec[i].copy()
        predictDefStr = predictionsDict[i]["str"]
        #cluster_runs_cmbn = funcH.append_to_vstack(cluster_runs_cmbn, predictionsDict[i]["prd"], dtype=int)
        print(predictDefStr, "Quality of cluster = {:6.4f}".format(quality_vec[i]), "number of clusters : ", kr_pdf_cur.shape)
        predictions_cur = predictionsDict[i]["prd"]
        unique_preds = np.unique(predictions_cur)

        kr_pdf_cur.sort_index(inplace=True)
        eci_N = np.array(eci_vec_cur * kr_pdf_cur['N'], dtype=float)
        eci_pd = pd.DataFrame(eci_vec_cur, columns=['ECi'])
        eci_N_pd = pd.DataFrame(eci_N, columns=['ECi_n'])
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd], axis=1)
        pd_comb.sort_values(by=['ECi_n', 'N'], inplace=True, ascending=[False, False])

        kr_pdf_FileName = "kluster_evaluations_" + predictDefStr + ".csv"
        kr_pdf_FileName = os.path.join(predictResultFold, kr_pdf_FileName)

        cols2add = np.zeros((clusterCounts[i], colCnt), dtype=float)
        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd], axis=1)

        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

        # pick first 10 15 20 25 samples according to lwca_mat
        for pi in range(0, clusterCounts[i]):
            cur_pred = unique_preds[pi]
            predictedSamples = funcH.getInds(predictions_cur, cur_pred)
            sampleLabels = labels_true[predictedSamples]
            lwca_cur = lwca_mat[predictedSamples, :]
            lwca_cur = lwca_cur[:, predictedSamples]
            simSum = np.sum(lwca_cur, axis=0) + np.sum(lwca_cur, axis=1).T
            v, idx = funcH.sortVec(simSum)
            sortedPredictionsIdx = predictedSamples[idx]
            sortedLabelIdx = labels_true[sortedPredictionsIdx]
            curSampleCntInCluster = len(sampleLabels)
            mappedClassOfKluster = funcH.get_most_frequent(list(sortedLabelIdx))

            for sj in range(0, colCnt):
                sCnt = sampleCntToPick[sj] if curSampleCntInCluster>sampleCntToPick[sj] else curSampleCntInCluster
                sampleLabelsPicked = sortedLabelIdx[:sCnt]
                purity_k, _, mappedClass = funcH.calcPurity(list(sampleLabelsPicked))
                if mappedClass == mappedClassOfKluster:
                    cols2add[pi, sj] = purity_k
                else:
                    cols2add[pi, sj] = -mappedClass+(mappedClassOfKluster/100)

        cols2add_pd = pd.DataFrame(cols2add, columns=columns)
        pd_comb = pd.concat([kr_pdf_cur, eci_pd, eci_N_pd, cols2add_pd], axis=1)
        pd_comb.sort_index(inplace=True)
        pd.DataFrame.to_csv(pd_comb, path_or_buf=kr_pdf_FileName)

def runScript_hmm(n_components = 30, transStepAllow = 15, n_iter = 1000, startModel = 'firstOnly', transitionModel = 'lr'):
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

# n_components = 30
# transStepAllow = 15
# n_iter = 1000
# startModel = 'decreasing'  # 'firstOnly' 'decreasing'
# transitionModel = 'lr'  # 'random' 'lr' 'circular'
# print("n_components({:d}),n_components({:d}),n_iter({:d}),startModel({}),transitionModel({})".format(n_components,transStepAllow,n_iter,startModel,transitionModel))
# labels_true, labels_pred, predHMM = runScript_hmm(n_components=n_components, transStepAllow=transStepAllow, n_iter=n_iter, startModel = startModel, transitionModel = transitionModel)
# labelNames = load_label_names()
#
# labels_true_nz, predHMM_nz = funcH.getNonZeroLabels(labels_true, predHMM)
# klusRet_nz_hmm, classRet_nz_hmm, _confMat_nz_hmm, c_pdf_nz_hmm, kr_pdf_nz_hmm = runForPred(labels_true_nz-1, predHMM_nz, labelNames, "hgsk256_KMeans_NZ_hmm")

# labels_true_nz, labels_pred_nz = funcH.getNonZeroLabels(labels_true, labels_pred)
# klusRet_nz, classRet_nz, _confMat_nz, c_pdf_nz, kr_pdf_nz = runForPred(labels_true_nz-1, labels_pred_nz, labelNames, "hgsk256_KMeans_NZ")

# labelNames.insert(0, "None")
# klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels_true, predHMM, labelNames, "hgsk256_KMeans_hmm")

# useNZ = True
# runScript01(useNZ)
# study01(useNZ)

# study02(ep=98)

#  useNZ = True
#  runScript01_next(useNZ)

for nos in [12]: #8, 10, 11, 12
    useNZ = True
    runScript01(useNZ, nos, rs=10)
    runScript01_next(useNZ, nos, rs=10)

# for nos in [10, 12]:
#     prs.run4All_createData(sign_countArr=[nos], dataToUseArr = ["hog", "skeleton", "sn"])
#     prs.createCombinedDatasets(numOfSigns=nos)
#     prs.runForBaseClusterResults(normMode='', numOfSignsArr=[nos])
#     prs.run4All_createData(sign_countArr=[nos], dataToUseArr=["hgsk"])
#     prs.runForBaseClusterResults(normMode='', numOfSignsArr=[nos], dataToUseArr=["hgsk"])
