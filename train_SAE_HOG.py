''' Code for Keyframe Detection,
	Authors : Doga Siyli and Batuhan Gundogdu
'''
from sys import argv
import numpy as np
import os
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
#%%
import modelFuncs as funcM
import dataLoaderFuncs as funcD
import helperFuncs as funcH
import time

## extra imports to set GPU options
################################### # TensorFlow wizardry 
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed 
config.gpu_options.allow_growth = True  
# Only allow a total of half the GPU memory to be allocated 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified. 
K.tensorflow_backend.set_session(tf.Session(config=config))

base_dir = funcH.getVariableByComputerName('base_dir')
data_dir = funcH.getVariableByComputerName('data_dir')
results_dir = funcH.getVariableByComputerName('results_dir')

posterior_dim = int(argv[1])# K number of clusters
weight_of_regularizer = float(argv[2]) # sparsity parametresi (a trade-off between reconstruction vs clustering)
applyCorr = 5
data_dim = 256  #PCA sonrası LBP dimension
exp_name = 'sae_p' + str(posterior_dim) + '_wr'+str(weight_of_regularizer)+ '_cor'+str(applyCorr)

csv_name = os.path.join(results_dir, 'epochs') + os.sep + exp_name + '.csv'
model_name = os.path.join(results_dir, 'models') + os.sep + exp_name + '.h5'
outdir = os.path.join(results_dir, 'results', exp_name)
funcH.createDirIfNotExist(os.path.join(results_dir, 'epochs'))
funcH.createDirIfNotExist(os.path.join(results_dir, 'models'))
funcH.createDirIfNotExist(outdir)

skipLoadOfOriginalData = True
if not skipLoadOfOriginalData:
    feat_set, labels_all, detailed_labels_all = funcD.loadData_hog(data_dir, loadHogIfExist=True, hogFeatsFileName='hog_set.npy', labelsFileName='labels.npy', detailedLabelsFileName='detailed_labels.npy')
else:
    feat_set = np.array([])
    labels_all = np.array([])
    detailed_labels_all = np.array([])

feat_set_pca, labels_all = funcD.applyPCA2Data(feat_set, labels_all, data_dir, data_dim, loadIfExist=True, pcaFeatsFileName='feat_set_pca.npy', labelsFileName='labels.npy')
non_zero_labels = labels_all[np.where(labels_all)]

#%%
batch_size = 16
epochs = 10 
embedding_dim = 128# deep model yapacaksak bunu kullanacağız

model, ES = funcM.createModel(data_dim, posterior_dim, weight_of_regularizer)
modelTest = funcM.createPredictModel(data_dim, posterior_dim)

checkpointer = ModelCheckpoint(filepath=model_name,verbose=0,save_best_only=False,period=1)
csv_logger = CSVLogger(csv_name,append=True, separator=';')
callbacks=[csv_logger,ES,checkpointer]

#%%
nmi_and_acc_file_name = outdir + os.sep + exp_name + '_nmi_acc.txt'
print('started training')

corrMode = False
if applyCorr>=2:
    neuralNetHandVideosFolder = os.path.join(base_dir, 'neuralNetHandVideos')
    corrFramesSignFileName = neuralNetHandVideosFolder + os.sep + 'corrFrames_All.npy'
    corrFramesAll = np.load(corrFramesSignFileName)
    inFeats = feat_set_pca[corrFramesAll[0, :], :]
    outFeats = feat_set_pca[corrFramesAll[1, :], :]
    #detailedLabelsFileNameFull = data_dir + os.sep + 'detailed_labels.npy'
    #detailedLabels_all = np.load(detailedLabelsFileNameFull)
    #non_zero_detailed_labels = detailed_labels_all[np.where(labels_all), :]
else:
    inFeats = feat_set_pca
    outFeats = feat_set_pca

for i in range(100):
    corrMode = applyCorr >= 2 and np.mod(i, applyCorr) == 0
    t = time.time()
    if corrMode:
        print('corrMode on')
        model.fit([inFeats],[outFeats],batch_size=batch_size,callbacks=[csv_logger,checkpointer],epochs=epochs,validation_split=0.0,shuffle=True,verbose=0)
    else:
        model.fit([feat_set_pca],[feat_set_pca],batch_size=batch_size,callbacks=[csv_logger,checkpointer],epochs=epochs,validation_split=0.0,shuffle=True,verbose=0)

    modelTest.load_weights(model_name, by_name=True)
    cluster_posteriors = np.transpose(modelTest.predict(feat_set_pca))
    predicted_labels = np.argmax(cluster_posteriors,axis=0)
    non_zero_predictions=predicted_labels[np.where(labels_all)]
    #nmi_cur = nmi(non_zero_labels,non_zero_predictions,average_method='geometric')
    #acc_cur = getAccFromConf(non_zero_labels, non_zero_predictions)
    elapsed_1 = time.time() - t
    nmi_cur, acc_cur = funcH.get_NMI_Acc(non_zero_labels, non_zero_predictions)

    f = open(nmi_and_acc_file_name, 'a+')
    f.write(' i =' + str(i) + ' NMI = ' + str(nmi_cur) + ' ACC= ' + str(acc_cur)+'\n')
    f.close()
    print(' i =', i, ' NMI = ', nmi_cur, ' ACC= ', acc_cur, '\n')

    elapsed_2 = time.time() - t

    print('elapsed_1(', elapsed_1, '), elapsed_2(', elapsed_2, ')')

#%%







