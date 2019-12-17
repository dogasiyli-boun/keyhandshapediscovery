import dataLoaderFuncs as funcD
import helperFuncs as funcH
import numpy as np
import os
import pandas as pd

labelNames = ["cat", "fish", "hen"]
labels_true   = np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3])-1
labels_pred_1 = np.array([1,1,1,1,1,1,1,1,4,4,2,2,2,2,2,2,5,5,3,3,3,3,3,3,3,3,3,3,5,6])-1
labels_pred_2 = np.array([1,1,1,1,1,1,7,4,1,2,2,2,2,2,2,2,5,5,3,3,3,3,3,3,3,3,3,3,5,6])-1
labels_pred_3 = np.array([1,1,1,1,1,1,4,4,4,2,2,2,2,2,2,2,5,5,3,3,3,3,3,3,3,3,3,3,3,3])-1

pd.set_option("display.precision", 3)

def runForPred(labels_true, labels_pred, labelNames):
    print("\r\n\r\n*-*-calcClusterMetrics-*-*\r\n\r\n")
    klusRet = funcH.calcClusterMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames)

    print("\r\n\r\n*-*-calcCluster2ClassMetrics-*-*\r\n\r\n")
    classRet, _confMat, c_pdf, kr_pdf = funcH.calcCluster2ClassMetrics(labels_true, labels_pred, removeZeroLabels=False, labelNames=labelNames)

    print("\r\n\r\n*-*-*-*-*\r\n\r\n")
    return klusRet, classRet, _confMat, c_pdf, kr_pdf

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

c_pdf = c_pdf_3[['class','%f1']].sort_index().rename(columns={"class": "f1Score", "%f1": "pred01"})
c_pdf.insert(2, "pred02", c_pdf_2[['%f1']].sort_index(), True)
c_pdf.insert(3, "pred03", c_pdf_3[['%f1']].sort_index(), True)
print(c_pdf)

