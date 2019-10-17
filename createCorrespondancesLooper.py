import dataLoaderFuncs as funcD
import helperFuncs as funcH
import os
import numpy as np

base_dir = funcH.getVariableByComputerName('base_dir')
data_dir = funcH.getVariableByComputerName('data_dir')
corrFramesAll, _ = funcD.loopTroughFeatureSet(base_dir=base_dir, data_dir=data_dir)
neuralNetHandVideosFolder = os.path.join(base_dir, 'neuralNetHandVideos')
corrFramesSignFileName = neuralNetHandVideosFolder + os.sep + 'corrFrames_All.npy'
np.save(corrFramesSignFileName, corrFramesAll)
print(corrFramesAll.shape)