import dataLoaderFuncs as funcD
import helperFuncs as funcH
import os
import numpy as np

base_dir = funcH.getVariableByComputerName('base_dir')
data_dir = funcH.getVariableByComputerName('data_dir')

hogFeatsFileName = 'hog_set_41.npy'
detailedLabelsFileName = 'detailed_labels_41.npy'
videosFolderName = 'neuralNetHandVideos_41'
corrFramesFileName = 'corrFrames_41.npy'

feat_set, labels_all, detailedLabels_all = funcD.loadData_hog(base_dir, loadHogIfExist=False, videosFolderName='neuralNetHandVideos_41', hogFeatsFileName='hog_set_41.npy', labelsFileName='labels_41.npy', detailedLabelsFileName='detailed_labels_41.npy')
corrFramesAll, _ = funcD.loopTroughFeatureSet(base_dir=base_dir, data_dir=data_dir,
                                             hogFeatsFileName=hogFeatsFileName,
                                             detailedLabelsFileName=detailedLabelsFileName,
                                             videosFolderName=videosFolderName)
neuralNetHandVideosFolder = os.path.join(base_dir, videosFolderName)
corrFramesSignFileName = neuralNetHandVideosFolder + os.sep + corrFramesFileName
np.save(corrFramesSignFileName, corrFramesAll)
print(corrFramesAll.shape)