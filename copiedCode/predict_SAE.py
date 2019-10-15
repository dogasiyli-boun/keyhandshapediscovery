''' Code for Keyframe Detection,
	Authors : Doga Siyli and Batuhan Gundogdu
'''
import numpy as np
import os
from keras.callbacks import CSVLogger, EarlyStopping,ModelCheckpoint,Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from keras.optimizers import Adam
from keras import backend as K
from sys import argv
from sklearn.decomposition import PCA

#%%

## extra imports to set GPU options
import tensorflow as tf
################################### # TensorFlow wizardry 
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed 
config.gpu_options.allow_growth = True  
# Only allow a total of half the GPU memory to be allocated 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified. 
K.tensorflow_backend.set_session(tf.Session(config=config))

def neg_l2_reg(activation):
	return -weight_of_regularizer*K.mean(K.square(activation))


data_dim = 256 # PCA sonrası LBP dimension
embedding_dim = 64# deep model yapacaksak bunu kullanacağız

batch_size = 256
weight_of_regularizer = 0.5 # sparsity parametresi (a trade-off between reconstruction vs clustering)
epochs = 1000 
posterior_dim = 64# K number of clusters

exp_name='sae_p'+str(posterior_dim)+'_wr'+str(weight_of_regularizer)

base_dir= os.path.abspath(os.path.join(os.getcwd(), os.pardir))

base_dir_train_feat = os.path.join(base_dir, 'train2')
outdir = os.path.join(base_dir, 'results',exp_name)

if not os.path.isdir(outdir):
    os.makedirs(outdir)
    

fileNames = os.listdir(base_dir_train_feat)
feat_set = np.array([0,0,0,0])
featCount = 0
for f in fileNames:
    if not f.startswith('.'):
        #read f as 'videoFeats_s001_v005' and get signID videoID
        feat_file = os.path.join(base_dir_train_feat,f)
        feat_current = np.loadtxt(feat_file)
        if np.all(feat_set==0):
            feat_set = feat_current
        else:              
            feat_set = np.vstack((feat_set, feat_current))

pca = PCA(n_components=data_dim)
feat_set_pca=pca.fit_transform(feat_set)
#%%
model = Sequential()
model.add(Dense(posterior_dim,activation='softmax',activity_regularizer=neg_l2_reg, input_shape=(data_dim,),name='encoder_layer'))
model.add(Dropout(0.5))
model.add(Dense(data_dim, activation='relu',use_bias=False, name='decoder_layer'))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001,clipnorm=0.1)

ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')

model.compile(loss='mean_squared_error',optimizer='adam') # if using DPGMM, use categorical_crossentropy

csv_name='../epochs/'+exp_name+'.csv'
model_name= '../models/'+exp_name+ '.h5'
checkpointer=ModelCheckpoint(filepath=model_name,verbose=0,save_best_only=True,period=1)
csv_logger = CSVLogger(csv_name,append=False, separator=';')
callbacks=[csv_logger,ES,checkpointer]
model.fit([feat_set_pca],[feat_set_pca],batch_size=batch_size,callbacks=[csv_logger,ES,checkpointer],epochs=epochs,validation_split=0.2,shuffle=True,verbose=1)

print('done training')