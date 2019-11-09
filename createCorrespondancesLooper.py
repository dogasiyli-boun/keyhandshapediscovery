import dataLoaderFuncs as funcD
import helperFuncs as funcH
import os
import numpy as np

base_dir = funcH.getVariableByComputerName('base_dir')
data_dir = funcH.getVariableByComputerName('data_dir')
numOfSigns = 41
dataToUse = 'hog'

videosFolderName = 'neuralNetHandVideos_' + str(numOfSigns)
hogFeatsFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='Data') # 'hogFeats_41.npy'
labelsFileName = funcD.getFileName(numOfSigns=numOfSigns, expectedFileType='Labels')  # 'labels_41.npy'
detailedLabelsFileName = funcD.getFileName(numOfSigns=numOfSigns, expectedFileType='DetailedLabels')  # 'detailedLabels_41.npy'
corrFramesFileName = funcD.getFileName(dataToUse=dataToUse, numOfSigns=numOfSigns, expectedFileType='CorrespendenceVec')  # 'hog_corrFrames_41.npy'

feat_set, labels_all, detailedLabels_all = funcD.loadData_hog(base_dir=base_dir, data_dir=data_dir,loadHogIfExist=True, numOfSigns=numOfSigns)
corrFramesAll, _ = funcD.getCorrespondentFrames(base_dir=base_dir, data_dir=data_dir, numOfSigns=numOfSigns, featType=dataToUse)
np.save(data_dir + os.sep + corrFramesFileName, corrFramesAll)
print(corrFramesAll.shape)