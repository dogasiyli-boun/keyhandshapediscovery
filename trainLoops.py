import datetime
import os
import time

import numpy as np

import dataLoaderFuncs as funcD
import helperFuncs as funcH


def updateNMIACCFile(epochID, nmi_cur, acc_cur, nmi_and_acc_file_name):
    if not os.path.isfile(nmi_and_acc_file_name):
        f = open(nmi_and_acc_file_name, 'w+')
        f.write('epochID*NMI*ACC\n')
        f.close()

    f = open(nmi_and_acc_file_name, 'a+')
    f.write(str(epochID) + ' * ' + str(nmi_cur) + ' * ' + str(acc_cur) + '\n')
    f.close()
    print(' epochID =', epochID, ' NMI = ', nmi_cur, ' ACC= ', acc_cur)

def savePredictedLabels(predictionLabelsDir, predicted_labels, epochID):
    predictedFileSaveAt = predictionLabelsDir + os.sep + 'predicted_labels' + str(epochID).zfill(3) + '.npy'
    np.save(predictedFileSaveAt, predicted_labels)

def getCorrespondanceIDs(sampleCount, trainParams, epochID):
    corr_indis_a = trainParams["corr_indis_a"]
    applyCorr = trainParams["applyCorr"]

    corrMode = applyCorr >= 1 and np.mod(epochID + 1, applyCorr) == 0
    if corrMode:
        corrFramesAll = trainParams["corrFramesAll"]
        if trainParams["corr_randMode"]:
            a_inds = np.random.randint(2, size=np.size(corrFramesAll[0], 1))
            col_idx = np.arange(np.size(corrFramesAll[0], 1))
            inIdx = corrFramesAll[0][a_inds, col_idx]
            outIdx = corrFramesAll[0][1 - a_inds, col_idx]
            corr_indis_a = a_inds[0:5]
        else:
            # switches between 0 and 1 if trainParams["corr_swapMode"] is true
            corr_indis_a = corr_indis_a if not trainParams["corr_swapMode"] else np.mod(corr_indis_a + 1, 2)
            inIdx = corrFramesAll[0][corr_indis_a, :]
            outIdx = corrFramesAll[0][1 - corr_indis_a, :]
            trainParams["corr_indis_a"] = corr_indis_a
        print('corrMode on, a_ind(', corr_indis_a, '), b_ind(', 1 - corr_indis_a, ')')
    else:
        print('corrMode off')
        inIdx = np.arange(sampleCount)
        outIdx = np.arange(sampleCount)

    return inIdx, outIdx, trainParams

def trainFramewise(trainParams, modelParams, model, modelTest, feat_set_pca, labels_all, directoryParams):
    epochFr = trainParams["epochFr"]
    epochTo = trainParams["epochTo"]
    predictionLabelsDir = directoryParams["predictionLabelsDir"]
    non_zero_labels = labels_all[np.where(labels_all)]
    sampleCount = feat_set_pca.shape[0]

    for epochID in range(epochFr, epochTo):
        t = time.time()

        # Prepare data for training
        inIdx, outIdx, trainParams = getCorrespondanceIDs(sampleCount, trainParams, epochID)
        inFeats = [np.array(feat_set_pca[inIdx, :]).squeeze()][0]
        outFeats = [np.array(feat_set_pca[outIdx, :]).squeeze()][0]

        # train
        print('*+*+*+* model fit started', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ', inFeats.shape', np.shape(inFeats), ', outFeats.shape',  np.shape(outFeats))
        model.fit(inFeats, outFeats, validation_split=0.0, shuffle=True, verbose=0,
                  batch_size=trainParams["batch_size"], callbacks=modelParams["callbacks"],
                  epochs=trainParams["subEpochs"])

        # load updated params into prediction model
        modelTest.load_weights(modelParams["model_name"], by_name=True)

        # predict
        cluster_posteriors = np.transpose(modelTest.predict(feat_set_pca))
        predicted_labels = np.argmax(cluster_posteriors, axis=0).T.squeeze()
        non_zero_predictions = predicted_labels[np.where(labels_all)]

        #
        elapsed = time.time() - t

        #
        nmi_cur, acc_cur = funcH.get_NMI_Acc(non_zero_labels, non_zero_predictions)

        # save progress
        savePredictedLabels(predictionLabelsDir, predicted_labels, epochID)
        updateNMIACCFile(epochID, nmi_cur, acc_cur, directoryParams["nmi_and_acc_file_name"])

        print('elapsedTime(', elapsed, '), ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def getRNNLabels_by_rnnDataMode(rnnDataMode, rnnParams, detailed_labels_all):
    if rnnDataMode==0:
        trainIDs = funcD.getRNNTrainLabels_lookBack(rnnParams["timesteps"], len(detailed_labels_all))
        predictIDs, frameIDsForLabelAcc = funcD.rnnPredictOverlappingDS(rnnParams["timesteps"], detailed_labels_all, verbose=0)
    elif rnnDataMode==1:
        trainIDs, trainLabels, trainLabels_detailed = \
            funcD.getRNNTrainLabels_patchPerVideos(
                rnnParams["timesteps"], rnnParams["patchFromEachVideo"], detailed_labels_all, verbose=0)
        predictIDs, frameIDsForLabelAcc = funcD.rnnPredictOverlappingDS(rnnParams["timesteps"], detailed_labels_all, verbose=0)
    elif rnnDataMode == 2:
        trainIDs = funcD.getRNNTrainLabels_frameOverlap(rnnParams["timesteps"], rnnParams["frameOverlap"], detailed_labels_all, verbose=0)
        predictIDs, frameIDsForLabelAcc = funcD.rnnPredictOverlappingDS(rnnParams["timesteps"], detailed_labels_all, verbose=0)
    return trainIDs, predictIDs, frameIDsForLabelAcc

def trainRNN(trainParams, modelParams, rnnParams, detailed_labels_all, model, modelTest, feat_set_pca, labels_all, directoryParams):
    epochFr = trainParams["epochFr"]
    epochTo = trainParams["epochTo"]
    predictionLabelsDir = directoryParams["predictionLabelsDir"]
    rnnDataMode = rnnParams["dataMode"]
    non_zero_labels = labels_all[np.where(labels_all)]
    sampleCount = feat_set_pca.shape[0]

    trainIDs, predictIDs, frameIDsForLabelAcc = getRNNLabels_by_rnnDataMode(rnnDataMode, rnnParams, detailed_labels_all)

    for epochID in range(epochFr, epochTo):
        t = time.time()

        # Prepare data for training
        if rnnDataMode==0 and trainParams["applyCorr"] >= 2:
            inIdx, outIdx, trainParams = getCorrespondanceIDs(sampleCount, trainParams, epochID)
            corrRNNMap = funcD.getRNNTrainLabels_lookBack(rnnParams["timesteps"], inIdx.shape[0])
            corr_a_inds = inIdx[corrRNNMap]
            corr_b_inds = outIdx[corrRNNMap]
            inFeats = funcD.rnnGetDataByTimeSteps(feat_set_pca, corr_a_inds, rnnParams["timesteps"])
            outFeats = funcD.rnnGetDataByTimeSteps(feat_set_pca, corr_b_inds, rnnParams["timesteps"])
        else:
            inFeats = funcD.rnnGetDataByTimeSteps(feat_set_pca, trainIDs, rnnParams["timesteps"])
            outFeats = inFeats

            # train
        print('*+*+*+* model fit started', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ', inFeats.shape', inFeats.shape, ', outFeats.shape', outFeats.shape)
        model.fit([inFeats], [outFeats], validation_split=0.0, shuffle=True, verbose=0,
                  batch_size=trainParams["batch_size"], callbacks=modelParams["callbacks"],
                  epochs=trainParams["subEpochs"])

        # load updated params into prediction model
        modelTest.load_weights(modelParams["model_name"], by_name=True)

        # prepare data for prediction
        inFeats = funcD.rnnGetDataByTimeSteps(feat_set_pca, predictIDs, rnnParams["timesteps"])

        # predict
        cluster_posteriors = modelTest.predict(inFeats)
        cp_in = cluster_posteriors.reshape(-1, modelParams["posterior_dim"])
        predicted_labels = np.argmax(cp_in, axis=1).T.squeeze()
        predicted_labels = predicted_labels[frameIDsForLabelAcc]
        non_zero_predictions = predicted_labels[np.where(labels_all)]

        #
        elapsed = time.time() - t

        #
        nmi_cur, acc_cur = funcH.get_NMI_Acc(non_zero_labels, non_zero_predictions)

        # save progress
        savePredictedLabels(predictionLabelsDir, predicted_labels, epochID)
        updateNMIACCFile(epochID, nmi_cur, acc_cur, directoryParams["nmi_and_acc_file_name"])

        print('elapsedTime(', elapsed, '), ended at ', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))