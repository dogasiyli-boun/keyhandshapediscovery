from keras.models import Model #Sequential
from keras.layers import Dense, Dropout, Input, GRU, SimpleRNN #LSTM, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import lossFuncs as funcL
import tensorflow as tf
from keras import backend as K

def createModel(data_dim, posterior_dim, weight_of_regularizer):
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


def createRNNModel(data_dim, posterior_dim, weight_of_regularizer, timesteps=5):
    hog_input = Input(shape=(timesteps, data_dim,))
    embedding = GRU(posterior_dim, activation='linear', return_sequences=True, stateful=False, name='first_rnn')(
        hog_input)
    clusters = Dense(posterior_dim, trainable=False, kernel_initializer='Identity', activation='softmax',
                     name='cluster_layer')(embedding)
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


def createRNNPredictModel(data_dim, posterior_dim, timesteps=5):
    test_input = Input(shape=(timesteps, data_dim,))
    test_embedding = GRU(posterior_dim, activation='linear', return_sequences=True, stateful=False, name='first_rnn')(
        test_input)
    test_clusters = Dense(posterior_dim, trainable=False, kernel_initializer='Identity', activation='softmax',
                          name='cluster_layer')(test_embedding)
    model = Model(inputs=test_input, outputs=test_clusters)
    return model
