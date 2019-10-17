''' Code for Keyframe Detection,
	Authors : Doga Siyli and Batuhan Gundogdu
'''
from sys import argv
import os
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import backend as K
import tensorflow as tf## extra imports to set GPU options
import numpy as np
import dataLoaderFuncs as funcD
import helperFuncs as funcH
import modelFuncs as funcM

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed 
config.gpu_options.allow_growth = True  
# Only allow a total of half the GPU memory to be allocated 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified. 
K.tensorflow_backend.set_session(tf.Session(config=config))

#%% some initial stuff
base_dir = funcH.getVariableByComputerName('base_dir')

posterior_dim = 64   #K number of clusters
weight_of_regularizer = 0.2    #sparsity parametresi (a trade-off between reconstruction vs clustering)
data_dim = 256    #PCA sonrası LBP dimension
exp_name = 'sae_p' + str(posterior_dim) + '_wr'+str(weight_of_regularizer)
csv_name = os.path.join(base_dir, 'epochs') + os.sep + 'exp_name' + '.csv'
model_name = os.path.join(base_dir, 'models') + os.sep + 'exp_name' + '.h5'

outdir = os.path.join(base_dir, 'results', exp_name)
funcH.createDirIfNotExist(outdir)
funcH.createDirIfNotExist(os.path.join(base_dir, 'epochs'))
funcH.createDirIfNotExist(os.path.join(base_dir, 'models'))

#%%
feat_set, labels_all, detailed_labels_all = funcD.loadData_hog(base_dir, loadHogIfExist=True, hogFeatsFileName='hog_set.npy', labelsFileName='labels.npy', detailedLabelsFileName='detailed_labels.npy')
feat_set_pca = funcD.applyPCA2Data(feat_set, base_dir, data_dim, loadIfExist=True, pcaFeatsFileName='feat_set_pca.npy')
non_zero_labels = labels_all[np.where(labels_all)]
non_zero_detailed_labels = detailed_labels_all[np.where(labels_all), :]

#%%
batch_size = 16
epochs = 100
embedding_dim = 64# deep model yapacaksak bunu kullanacağız

model, ES = funcM.createModel(data_dim, posterior_dim, weight_of_regularizer)
checkpointer = ModelCheckpoint(filepath=model_name, verbose=0, save_best_only=False, period=1)
csv_logger = CSVLogger(csv_name, append=False, separator=';')
callbacks=[csv_logger, ES, checkpointer]

modelTest = funcM.createPredictModel(data_dim, posterior_dim)

# %%
nmi_and_acc_file_name  = '../'+exp_name + 'nmi_acc.txt'
print('started training')
for i in range(100):
    model.fit([feat_set_pca],[feat_set_pca], batch_size=batch_size, callbacks=[csv_logger,checkpointer], epochs=epochs, validation_split=0.0, shuffle=True, verbose=0)

    modelTest.load_weights(model_name, by_name=True)

    cluster_posteriors = np.transpose(modelTest.predict(feat_set_pca))
    predicted_labels = np.argmax(cluster_posteriors, axis=0)
    non_zero_predictions=predicted_labels[np.where(labels_all)]

    nmi_cur, acc_cur = funcH.get_NMI_Acc(non_zero_labels, non_zero_predictions)

    f= open(nmi_and_acc_file_name,'a+')
    f.write(' i =' + str(i) + ' NMI = ' + str(nmi_cur) + ' ACC= ' + str(acc_cur)+'\n')
    #f.write('r*i(', i, '), NMI = ', nmi_cur, ', acc = ', acc_cur,'\n')
    f.close()

print('done training')