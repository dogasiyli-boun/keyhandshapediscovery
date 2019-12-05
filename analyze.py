import os
import numpy as np
import matplotlib.pyplot as plt
import helperFuncs as funcH

"""
========================================
Create 2D bar graphs in different planes
========================================

Demonstrates making a 3D plot which has 2D bar graphs projected onto
planes y=0, y=1, etc.
"""

def runExampleFunc():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.random.rand(20)

        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def analyzeResultByFolder():
    labelsFileNameFull = '/home/doga/DataFolder/neuralNetHandVideos' + os.sep + 'labels_convModel.npy'
    clusterFolderNameBase = '/home/doga/Desktop/neuralNetHandImages/labelSaveFolder_resnet18'
    labels = ['urgent01', 'urgent02', 'singlePalmUp', 'noseFist', 'cShow', 'swipe', 'ok', 'headShow', 'showFiveBack',
              'bowlShow', 'niceShow', 'phoneTwist', 'shoulderShow', 'hipShow', 'showFiveOpen', 'claw', 'meShow',
              'vaccinateOpen', 'vaccinateClose', 'tapOnHand', 'sideFive', 'frustrated', 'sickShow', 'oneShow',
              'likeThis', 'fist']

    completedEpochCnt = funcH.numOfFilesInFolder(clusterFolderNameBase, startswith="labels", endswith=".npz")
    for iterID in range(completedEpochCnt):
        clusterFileName = clusterFolderNameBase + os.sep + 'labels_{:03d}.npz'.format(iterID+1)
        fName_clusterCnt_sorted = clusterFolderNameBase + os.sep + 'clusSorted_{:03d}.png'.format(iterID+1)
        fName_conf_normNot = clusterFolderNameBase + os.sep + 'confNormNot_{:03d}.png'.format(iterID+1)
        fName_conf_normRow = clusterFolderNameBase + os.sep + 'confNormRow_{:03d}.png'.format(iterID+1)
        fName_conf_normCol = clusterFolderNameBase + os.sep + 'confNormCol_{:03d}.png'.format(iterID+1)
        clusterResults = np.load(clusterFileName)
        print(clusterResults.files)
        labelsTrInit = clusterResults['labelsTrInit']
        predClusters = clusterResults['predClusters']
        predictionsTr = clusterResults['predictionsTr']
        acc_lab = clusterResults['acc_lab']
        acc_lab_nonzero = clusterResults['acc_lab_nonzero']

        nonZeroLabs = labelsTrInit[np.where(labelsTrInit)]
        nonZeroPred = predClusters[np.where(labelsTrInit)]

        predClustCnt = np.unique(nonZeroPred)
        num_bins = len(predClustCnt)
        n, bins, patches = plt.hist(nonZeroPred, num_bins, facecolor='blue', alpha=0.5)
        plt.clf()

        sortedN, _ = funcH.sortVec(n)
        y_pos = np.arange(len(sortedN))
        plt.bar(y_pos, sortedN, align='center', alpha=0.5)
        plt.ylim(0,1000)
        plt.xlim(0,256)
        #plt.savefig(fName_clusterCnt_sorted)

        _confMat, kluster2Classes = funcH.confusionFromKluster(nonZeroLabs, nonZeroPred)
        funcH.plotConfMat(_confMat, labels, saveFileName=fName_conf_normNot, iterID=iterID)
        funcH.plotConfMat(_confMat, labels, saveFileName=fName_conf_normCol, iterID=iterID, normalizeByAxis=0, add2XLabel='normalized')
        funcH.plotConfMat(_confMat, labels, saveFileName=fName_conf_normRow, iterID=iterID, normalizeByAxis=1, add2YLabel='normalized')

        #n, bins, _ = plt.hist(kluster2Classes, bins=np.unique(kluster2Classes), facecolor='red', alpha=0.5)
        #sortedClusterCounts, idx = funcH.sortVec(n)
        #y_pos = np.arange(len(sortedClusterCounts))
        #plt.clf()
        #plt.bar(y_pos[idx], sortedClusterCounts, align='center', alpha=0.5)
        #plt.xlabel(labels)
        #plt.show()

        #x = 5