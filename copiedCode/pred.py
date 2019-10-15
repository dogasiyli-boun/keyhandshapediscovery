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
def create_dataset(dataset, look_back=100):
	dataX= []
	for i in range(len(dataset)/look_back):
		a = dataset[i*look_back:((i+1)*look_back), :]
		dataX.append(a)
	return np.array(dataX)

def neg_l2_reg(activation):
	return -weight_of_regularizer*K.mean(K.square(activation))

def printUsage():
	msg='Usage : pred.py ver_name embedding_size posterior_size sparsity_lambda'
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
model_name= 'models/'+exp_name+ '.h5'

embedding_model = Sequential()
embedding_model.add(LSTM(embedding_dim, return_sequences=True, stateful=False, input_shape=(timesteps,data_dim,),name='first_rnn'))
embedding_model.add(Dense(posterior_dim, activation='softmax',use_bias=False, activity_regularizer=neg_l2_reg, name='softmax_layer'))
embedding_model.load_weights(model_name, by_name=True)


test_data = np.load(base_dir_test_feat+'data.npy')
test_data = test_data[:,0:data_dim]
offsets = np.load(base_dir_test_feat+'offsets_data.npy')

seg_test_data={}
beg=0
for i in range(len(offsets)):
    seg_test_data[i] = test_data[beg:offsets[i]]
    beg=offsets[i]
seg_test_data[i+1] = test_data[offsets[i]:]

padded_sizes={}
padded_matrices = {}

for i in range(len(offsets)+1):
    padded_sizes[i]= ((seg_test_data[i].shape[0]/timesteps)+1)*timesteps - seg_test_data[i].shape[0]
    padder = np.zeros((padded_sizes[i],data_dim))
    padded_matrices[i]= np.concatenate((seg_test_data[i],padder))
    padded_matrices[i] = np.reshape(padded_matrices[i],(padded_matrices[i].shape[0]/timesteps,timesteps,data_dim))
    


keys = []
with open(base_dir_test_feat+'keys_data.txt') as file:
    for line in file:
        line=line.strip()
        #line=re.split('\s+',line)
        keys.append(line)
pred_dir = outdir+'english/test/' 
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
for keyword in padded_matrices:
    decoded = embedding_model.predict(padded_matrices[keyword],batch_size=1)
    decoded = np.reshape(decoded,(decoded.shape[0]*timesteps,posterior_dim))
    decoded = decoded[:decoded.shape[0]-padded_sizes[keyword]]
    filename = keys[keyword]   
    f = open(pred_dir + filename[:-4] + '.txt', 'a')
    for line in range(len(decoded)):
        for feature in range(posterior_dim-1):
            tmp = decoded[line][feature]
            f.write("%4f " % tmp)
        tmp = decoded[line][posterior_dim-1]
        f.write("%4f" % tmp)
        f.write('\n')
