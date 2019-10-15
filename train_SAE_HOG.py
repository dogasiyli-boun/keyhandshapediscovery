''' Code for Keyframe Detection,
	Authors : Doga Siyli and Batuhan Gundogdu
'''
import numpy as np
import os
from keras.callbacks import CSVLogger, EarlyStopping,ModelCheckpoint,Callback
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense,Dropout,Activation, Input
from keras.optimizers import Adam
from keras import backend as K
from sys import argv
from sklearn.decomposition import PCA
from skimage import data
from skimage.feature import hog
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


def neg_l2_reg2(activation,weight_of_regularizer):
	return -weight_of_regularizer*(K.square(activation))

def penalized_loss(l2_val):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))+K.mean(l2_val)
    return loss
def only_sparsity_loss(l2_val):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))*0+K.mean(l2_val)
    return loss
      
#%% loading data stuff
def loadData(base_dir, loadHogIfExist = True, hogFeatsFileName = 'hog_set.npy'):
    base_dir_train_feat = os.path.join(base_dir, 'neuralNetHandVideos')    
    
    if loadHogIfExist and os.path.isfile(hogFeatsFileName):
        print('loading exported feat_set from(', hogFeatsFileName, ')')
        feat_set = np.load(hogFeatsFileName)
        print('loaded exported feat_set(',  feat_set.shape, ') from(', hogFeatsFileName, ')')
    else:
        feat_set = np.array([0,0,0,0])
        foldernames = os.listdir(base_dir_train_feat)
        for f in foldernames:
                sign_folder=os.path.join(base_dir_train_feat,str(f).format(':02d'))
                videos=os.listdir(sign_folder)
                print(f)
                for v in videos:
                    video_folder=os.path.join(sign_folder,v)
                    frames=os.listdir(video_folder)
                    for frame in sorted(frames):
                        if frame.endswith('.png'):
                            frame_name=os.path.join(video_folder,frame)
                            img=data.load(frame_name)
                            feat_current = hog(img,pixels_per_cell=(32, 32), cells_per_block=(4, 4))
                            if np.all(feat_set==0):
                                feat_set = feat_current
                            else:              
                                feat_set = np.vstack((feat_set, feat_current))
        print('saving exported feat_set(',  feat_set.shape, ') into(', hogFeatsFileName, ')')
        np.save(hogFeatsFileName, feat_set) 
    return feat_set

def applyPCA2Data(feat_set, base_dir, loadIfExist = True, pcaFeatsFileName = 'feat_set_pca.npy'):
    pcaFeatsFileName = 'feat_set_pca.npy'
    if loadIfExist and os.path.isfile(pcaFeatsFileName):
        print('loading feat_set_pca from(', pcaFeatsFileName, ')')
        feat_set_pca = np.load(pcaFeatsFileName)
        print('loaded feat_set_pca(',  feat_set_pca.shape, ') from(', pcaFeatsFileName, ')')        
    else:
        feat_set = loadData(base_dir)
        pca = PCA(n_components=data_dim)
        feat_set_pca=pca.fit_transform(feat_set)
        np.save('feat_set_pca.npy', feat_set_pca)
    return feat_set_pca

def createModel(data_dim, posterior_dim, weight_of_regularizer):
    hog_input = Input(shape=(data_dim,))
    embedding = Dense(posterior_dim, input_shape=(data_dim,),name='encoder_layer')(hog_input)
    dropped_embedding = Dropout(0.5)(embedding)
    clusters = Dense(posterior_dim,activation='softmax',trainable=False,kernel_initializer='Identity',name='cluster_layer')(dropped_embedding)
    sparsity_value = neg_l2_reg2(clusters,weight_of_regularizer)
    decoded = Dense(data_dim,use_bias=False, name='decoder_layer')(clusters)
    model = Model(inputs=hog_input,outputs=decoded)
    adam_this = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001,clipnorm=0.1)
    ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')
    model.compile(loss=penalized_loss(l2_val=sparsity_value), metrics = [only_sparsity_loss(l2_val=sparsity_value),'mse'], optimizer='adam_this') # if using DPGMM, use categorical_crossentropy
    return model, ES

#%% some initial stuff
base_dir= os.path.abspath(os.path.join(os.getcwd(), os.pardir))

posterior_dim = 64# K number of clusters
weight_of_regularizer = 0.5 # sparsity parametresi (a trade-off between reconstruction vs clustering)
data_dim = 256 # PCA sonrası LBP dimension
exp_name='sae_p'+str(posterior_dim)+'_wr'+str(weight_of_regularizer)
csv_name='../epochs/'+exp_name+'.csv'
model_name= '../models/'+exp_name+ '.h5'

outdir = os.path.join(base_dir, 'results',exp_name)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    
#%%
feat_set = loadData(base_dir, loadHogIfExist = True, hogFeatsFileName = 'hog_set.npy')
feat_set_pca = applyPCA2Data(feat_set, base_dir, loadIfExist = True, pcaFeatsFileName = 'feat_set_pca.npy')

#%%
#model = Sequential()
#model.add(Dense(posterior_dim, input_shape=(data_dim,),name='encoder_layer'))
#model.add(Dropout(0.5))
#model.add(Dense(posterior_dim,activity_regularizer=neg_l2_reg,activation='softmax',trainable=False,kernel_initializer='Identity',name='cluster_layer'))
#model.add(Dense(data_dim,use_bias=False, name='decoder_layer'))
batch_size = 16
epochs = 1000 
embedding_dim = 64# deep model yapacaksak bunu kullanacağız

model, ES = createModel(data_dim, posterior_dim, weight_of_regularizer)
checkpointer=ModelCheckpoint(filepath=model_name,verbose=0,save_best_only=True,period=1)
csv_logger = CSVLogger(csv_name,append=False, separator=';')
callbacks=[csv_logger,ES,checkpointer]
model.fit([feat_set_pca],[feat_set_pca],batch_size=batch_size,callbacks=[csv_logger,ES,checkpointer],epochs=epochs,validation_split=0.2,shuffle=True,verbose=1)

print('done training')