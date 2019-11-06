from keras.models import Model #Sequential
from keras.layers import Dense, Dropout, Input, GRU, SimpleRNN #LSTM, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import lossFuncs as funcL
import tensorflow as tf
from keras import backend as K

def createModel(data_dim, modelParams):
    posterior_dim = modelParams["posterior_dim"]
    weight_of_regularizer = modelParams["weight_of_regularizer"]

    hog_input = Input(shape=(data_dim,))
    embedding = Dense(posterior_dim, input_shape=(data_dim,), name='encoder_layer')(hog_input)
    dropped_embedding = Dropout(0.5)(embedding)
    clusters = Dense(posterior_dim, activation='softmax', trainable=False, kernel_initializer='Identity', name='cluster_layer')(dropped_embedding)
    sparsity_value = funcL.neg_l2_reg2(clusters, weight_of_regularizer)
    decoded = Dense(data_dim, use_bias=False, name='decoder_layer')(clusters)
    model = Model(inputs=hog_input,outputs=decoded)
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001,clipnorm=0.1)
    ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')
    model.compile(loss=funcL.penalized_loss(l2_val=sparsity_value), metrics=[funcL.only_sparsity_loss(l2_val=sparsity_value),'mse'], optimizer='adam') # if using DPGMM, use categorical_crossentropy
    return model, ES

def createPredictModel(data_dim, posterior_dim):
    test_input = Input(shape=(data_dim,))
    test_embedding = Dense(posterior_dim, input_shape=(data_dim,), name='encoder_layer')(test_input)
    test_clusters = Dense(posterior_dim, activation='softmax', trainable=False, kernel_initializer='Identity', name='cluster_layer')(test_embedding)
    model = Model(inputs=test_input, outputs=test_clusters)
    return model

def createRNNModel(data_dim, modelParams, rnnParams):
    posterior_dim = modelParams["posterior_dim"]
    weight_of_regularizer = modelParams["weight_of_regularizer"]
    timesteps = rnnParams["timesteps"]

    hog_input = Input(shape=(timesteps, data_dim,))
    embedding = GRU(posterior_dim, activation='linear', return_sequences=True, stateful=False, name='first_rnn')(hog_input)
    clusters = Dense(posterior_dim, trainable=False, kernel_initializer='Identity', activation='softmax', name='cluster_layer')(embedding)
    sparsity_value = funcL.neg_l2_reg2(clusters, weight_of_regularizer)
    decoded = Dense(data_dim, use_bias=False, name='decoder_layer')(clusters)
    # decoded = SimpleRNN(data_dim, return_sequences=True, stateful=False, name='second_rnn')(clusters)

    model = Model(inputs=hog_input, outputs=decoded)
    # adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001,clipnorm=0.1)
    ES = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=0, mode='auto')
    model.compile(loss=funcL.penalized_loss(l2_val=sparsity_value),
                  metrics=[funcL.only_sparsity_loss(l2_val=sparsity_value), 'mse'],
                  optimizer='adam')  # if using DPGMM, use categorical_crossentropy
    return model, ES

def createRNNPredictModel(data_dim, modelParams, rnnParams):
    posterior_dim = modelParams["posterior_dim"]
    timesteps = rnnParams["timesteps"]

    test_input = Input(shape=(timesteps, data_dim,))
    test_embedding = GRU(posterior_dim, activation='linear', return_sequences=True, stateful=False, name='first_rnn')(test_input)
    test_clusters = Dense(posterior_dim, trainable=False, kernel_initializer='Identity', activation='softmax', name='cluster_layer')(test_embedding)
    model = Model(inputs=test_input, outputs=test_clusters)
    return model

def getModels(data_dim, modelParams, rnnParams):
    if modelParams["trainMode"] == "sae" or modelParams["trainMode"] == "cosae":
        model, ES = createModel(data_dim, modelParams=modelParams)
        modelTest = createPredictModel(data_dim, posterior_dim=modelParams["posterior_dim"])
    elif modelParams["trainMode"] == "corsa":
        model, ES = createRNNModel(data_dim, modelParams=modelParams, rnnParams=rnnParams)
        modelTest = createRNNPredictModel(data_dim, modelParams=modelParams, rnnParams=rnnParams)
    return model, modelTest, ES

def initialize_RNN_Parameters(valuesParamsCur, dvSetParamsCur):
    if valuesParamsCur["trainMode"] == 'corsa':
        rnnDataMode = valuesParamsCur["rnnDataMode"]
        rnnTimesteps = valuesParamsCur["rnnTimesteps"]
        rnnPatchFromEachVideo = valuesParamsCur["rnnPatchFromEachVideo"]
        rnnFrameOverlap = valuesParamsCur["rnnFrameOverlap"]

        rnnTimesteps_dvSet = dvSetParamsCur["rnnTimesteps"]
        rnnPatchFromEachVideo_dvSet = dvSetParamsCur["rnnPatchFromEachVideo"]
        rnnFrameOverlap_dvSet = dvSetParamsCur["rnnFrameOverlap"]

        if rnnDataMode == 1 and rnnPatchFromEachVideo < 0:
            rnnPatchFromEachVideo = 10
            rnnPatchFromEachVideo_dvSet = True
            print('rnnPatchFromEachVideo will be set to 10 due to trainMode(corsa) and rnnDataMode(1)')
        if rnnDataMode == 2 and rnnFrameOverlap < 0:
            rnnFrameOverlap = 5
            rnnFrameOverlap_dvSet = True
            print('rnnFrameOverlap will be set to 5 due to trainMode(corsa) and rnnDataMode(2)')
        if rnnTimesteps < 0:
            if rnnDataMode == 0:
                rnnTimesteps = 1
            elif rnnDataMode == 1:
                rnnTimesteps = 10
            elif rnnDataMode == 2:
                rnnTimesteps = 10
            rnnTimesteps_dvSet = True
            print('rnnTimesteps will be set to ', str(rnnTimesteps), ' due to rnnDataMode(', str(rnnDataMode), ')')

        dvSetParamsCur["rnnTimesteps"] = rnnTimesteps_dvSet
        dvSetParamsCur["rnnPatchFromEachVideo"] = rnnPatchFromEachVideo_dvSet
        dvSetParamsCur["rnnFrameOverlap"] = rnnFrameOverlap_dvSet
        valuesParamsCur["rnnTimesteps"] = rnnTimesteps
        valuesParamsCur["rnnPatchFromEachVideo"] = rnnPatchFromEachVideo
        valuesParamsCur["rnnFrameOverlap"] = rnnFrameOverlap
    return valuesParamsCur, dvSetParamsCur