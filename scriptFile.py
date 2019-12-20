import dataLoaderFuncs as funcD
import helperFuncs as funcH
import numpy as np
import os
import pandas as pd
import Cluster_Ensembles as CE

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

    klusRet_1, classRet_1, _confMat_1, c_pdf_1, kr_pdf_1 = runForPred(labels_true, labels_pred_1, labelNames)
    klusRet_2, classRet_2, _confMat_2, c_pdf_2, kr_pdf_2 = runForPred(labels_true, labels_pred_2, labelNames)
    klusRet_3, classRet_3, _confMat_3, c_pdf_3, kr_pdf_3 = runForPred(labels_true, labels_pred_3, labelNames)

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

def runForPred(labels_true, labels_pred, labelNames, predictDefStr):
    print("\r\n*-*-*start-", predictDefStr, "-end*-*-*\r\n")
    print("\r\n\r\n*-*-", predictDefStr, "calcClusterMetrics-*-*\r\n\r\n")
    klusRet = funcH.calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames)

    print("\r\n\r\n*-*-", predictDefStr, "calcCluster2ClassMetrics-*-*\r\n\r\n")
    classRet, _confMat, c_pdf, kr_pdf = funcH.calcCluster2ClassMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames, predictDefStr=predictDefStr)

    print("*-*-*end-", predictDefStr, "-end*-*-*\r\n")
    return klusRet, classRet, _confMat, c_pdf, kr_pdf

def loadBaseResult(fileName):
    results_dir = funcH.getVariableByComputerName('results_dir')
    preds = np.load(os.path.join(results_dir, 'baseResults', fileName + '.npz'))
    labels_true = np.asarray(preds['arr_0'], dtype=int)
    labels_pred = np.asarray(preds['arr_1'], dtype=int)
    return labels_true, labels_pred

def loadLabelsAndPreds(useNZ):
    data_dir = funcH.getVariableByComputerName('data_dir')
    results_dir = funcH.getVariableByComputerName('results_dir')
    labelnames_csv_filename = os.path.join(data_dir, "khsList_0_11_26.csv")
    labelNames = list(pd.read_csv(labelnames_csv_filename, sep=",")['name'].values.flatten())

    pred01Str = "hgskKmeans" + ("_nz" if useNZ else "")
    pred02Str = "hgskCosae" + ("_nz" if useNZ else "")
    pred03Str = "sn256Kmeans" + ("_nz" if useNZ else "")
    pred04Str = "sn256Cosae" + ("_nz" if useNZ else "")

    pred02_fname = os.path.join(results_dir, "results", "cosae_pd256_wr1.0_hgsk256_11_bs16_rs1_cp2_cRM0", "predicted_labels001.npy")
    pred04_fname = os.path.join(results_dir, "results", "cosae_pd256_wr1.0_sn256_11_bs16_rs1_cp2_cRM0", "predicted_labels049.npy")

    labels_true, labels_pred_1 = loadBaseResult("hgsk256_11_KMeans_256")
    labels_pred_2 = np.load(pred02_fname)
    _, labels_pred_3 = loadBaseResult("sn256_11_KMeans_256")
    labels_pred_4 = np.load(pred04_fname)

    print(labelNames, "\r\n", labels_true, "\r\n", labels_pred_1)

    labels_true_nz, labels_pred_nz_1 = funcH.getNonZeroLabels(labels_true, labels_pred_1)
    _, labels_pred_nz_2 = funcH.getNonZeroLabels(labels_true, labels_pred_2)
    _, labels_pred_nz_3 = funcH.getNonZeroLabels(labels_true, labels_pred_3)
    _, labels_pred_nz_4 = funcH.getNonZeroLabels(labels_true, labels_pred_4)

    if useNZ:
        labels_pred_1 = labels_pred_nz_1
        labels_pred_2 = labels_pred_nz_2
        labels_pred_3 = labels_pred_nz_3
        labels_pred_4 = labels_pred_nz_4
        labels_true = labels_true_nz-1
    else:
        labelNames.insert(0, "None")


    print(pred01Str)
    print(pred02Str)
    print(pred03Str)
    print(pred04Str)
    predictionsDict = []
    predictionsDict.append({"str": pred01Str, "prd": labels_pred_1})
    predictionsDict.append({"str": pred02Str, "prd": labels_pred_2})
    predictionsDict.append({"str": pred03Str, "prd": labels_pred_3})
    predictionsDict.append({"str": pred04Str, "prd": labels_pred_4})
    N = 4

    return labelNames, labels_true, predictionsDict, N

def setPandasDisplayOpts():
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option("display.precision", 3)

def runScript01(useNZ):
    setPandasDisplayOpts()

    labelNames, labels_true, predictionsDict, N = loadLabelsAndPreds(useNZ)
    print(predictionsDict[0]["str"])
    print(predictionsDict[1]["prd"])

    cluster_runs = predictionsDict[0]["prd"]
    for i in range(1, N):
        cluster_runs = np.asarray(np.vstack((cluster_runs, predictionsDict[i]["prd"])), dtype=int)
    consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose=False, N_clusters_max=256)

    predCombined = "combined" + ("_nz" if useNZ else "")
    predictionsDict.append({"str": predCombined, "prd": consensus_clustering_labels})

    resultsDict = []
    for i in range(0, N+1):
        klusRet, classRet, _confMat, c_pdf, kr_pdf = runForPred(labels_true, predictionsDict[i]["prd"], labelNames, predictionsDict[i]["str"])
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

useNZ = True
runScript01(useNZ)