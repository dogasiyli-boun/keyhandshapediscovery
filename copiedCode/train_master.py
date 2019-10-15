''' Code for Posteriti,
	Authors : Alican Gok and Batuhan Gundogdu
'''
import numpy as np
import os
from keras.callbacks import CSVLogger, EarlyStopping,ModelCheckpoint,Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import backend as K
from sys import argv

## extra imports to set GPU options
import tensorflow as tf
################################### # TensorFlow wizardry 
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed 
config.gpu_options.allow_growth = True  
# Only allow a total of half the GPU memory to be allocated 
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified. 
k.tensorflow_backend.set_session(tf.Session(config=config))


def create_dataset(dataset, look_back=100):
	dataX= []
	for i in range(len(dataset)/look_back):
		a = dataset[i*look_back:((i+1)*look_back), :]
		dataX.append(a)
	return np.array(dataX)

def neg_l2_reg(activation):
	return -weight_of_regularizer*K.mean(K.square(activation))

def printUsage():
	msg='Usage : train.py ver_name embedding_size posterior_size sparsity_lambda'
	print(msg)

try:
	_ , exp_name, embedding_dim, posterior_dim, sparsity_lambda = argv

except ValueError as e:
	printUsage()
	exit()

data_dim = 13
embedding_dim = int(embedding_dim)
timesteps = 100
batch_size = 256
weight_of_regularizer = float(sparsity_lambda)
epochs = 1000 
posterior_dim = int(posterior_dim)

base_dir = os.getcwd()
base_dir_train_feat = base_dir + '/train/'
base_dir_test_feat = base_dir + '/test/'
outdir = base_dir + '/Results/'+exp_name+'/'

data = np.load(base_dir_train_feat+'data.npy')
data = data[:,0:13]
train_data = create_dataset(data)

model = Sequential()
model.add(LSTM(embedding_dim, return_sequences=True, stateful=False, input_shape=(timesteps,data_dim,),name='first_rnn'))

model.add(Dense(posterior_dim, activation='softmax',use_bias=False, activity_regularizer=neg_l2_reg, name='softmax_layer'))

model.add(Dense(embedding_dim, activation='relu',use_bias=False, name='decoder_layer'))
model.add(LSTM(data_dim, return_sequences=True, stateful=False))

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001,clipnorm=0.1)

ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=0, mode='auto')

model.compile(loss='mean_squared_error',optimizer='adam') # if using DPGMM, use categorical_crossentropy

csv_name='epochs/'+exp_name+'.csv'
model_name= 'models/'+exp_name+ '.h5'
checkpointer=ModelCheckpoint(filepath=model_name,verbose=0,save_best_only=True,period=1)
csv_logger = CSVLogger(csv_name,append=False, separator=';')
model.fit([train_data],[train_data],batch_size=batch_size, callbacks=[csv_logger,ES,checkpointer],epochs=epochs,validation_split=0.2,shuffle=True,verbose=0)

print('done training')
