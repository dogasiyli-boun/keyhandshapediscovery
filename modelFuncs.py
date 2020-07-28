from keras.models import Model #Sequential
from keras.layers import Dense, Dropout, Input, GRU, SimpleRNN #LSTM, Activation
from keras.callbacks import EarlyStopping
import lossFuncs as funcL

from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
from torch.nn import Dropout as torch_do
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.nn import Softmax
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import save as torch_save
from torch import load as torch_load

import numpy as np
from sklearn.metrics import accuracy_score

def create_hidstate_dict(hid_state_cnt_vec, init_mode_vec = None, act_vec=None, verbose=0):
    hid_state_cnt = len(hid_state_cnt_vec)
    hidStatesDict = {}
    for i in range(hid_state_cnt):
        hid_state_id_str = str(i+1).zfill(2)
        hid_state_name = "hidStateDict_" + hid_state_id_str
        dim_str = str(hid_state_cnt_vec[i])
        dim_int = int(hid_state_cnt_vec[i])
        try:#if init_mode_vec is not None and len(init_mode_vec)>=i:
            initMode = init_mode_vec[i]
        except:#else:
            initMode = "kaiming_uniform_"
        try:#if act_vec is not None and len(act_vec)>=i:
            actStr = act_vec[i]
        except:#else:
            actStr = "relu"
        if verbose>0:
            print(hid_state_name, ' = {"dimOut": "', dim_str, '", "initMode": "', initMode ,'", "act": "', actStr,'"}')
        hidStatesDict[hid_state_id_str] = {"dimOut": dim_int, "initMode": initMode, "act": actStr}
    return hidStatesDict

#nice one
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
    if rnnParams["dropout"] > 0:
        embedding = Dropout(0.5)(embedding)
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
    elif modelParams["trainMode"] == "rsa" or modelParams["trainMode"] == "corsa":
        model, ES = createRNNModel(data_dim, modelParams=modelParams, rnnParams=rnnParams)
        modelTest = createRNNPredictModel(data_dim, modelParams=modelParams, rnnParams=rnnParams)
    return model, modelTest, ES

def initialize_RNN_Parameters(valuesParamsCur, dvSetParamsCur):
    if valuesParamsCur["trainMode"] == 'rsa':
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
            print('rnnPatchFromEachVideo will be set to 10 due to trainMode(rsa) and rnnDataMode(1)')
        if rnnDataMode == 2 and rnnFrameOverlap < 0:
            rnnFrameOverlap = 5
            rnnFrameOverlap_dvSet = True
            print('rnnFrameOverlap will be set to 5 due to trainMode(rsa) and rnnDataMode(2)')
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
    elif valuesParamsCur["trainMode"] == 'corsa':
        rnnDataMode = valuesParamsCur["rnnDataMode"]
        assert (rnnDataMode == 0), "rnnDataMode(" + str(rnnDataMode) + ") must be 0"

    return valuesParamsCur, dvSetParamsCur

class MLP(Module):
    # define model elements
    def __init__(self, dim_of_input, hidCounts, classCount):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(dim_of_input, hidCounts[0])
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(hidCounts[0], hidCounts[1])
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(hidCounts[1], classCount)
        xavier_uniform_(self.hidden3.weight)
        #self.act3 = Softmax(dim=1)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        #X = self.act3(X)
        return X

    # train the model
    def train_model(self, train_dl, epochCnt=500):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(epochCnt):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

    # evaluate the model
    def evaluate_model(self, test_dl):
        predictions, actuals = list(), list()
        sm = Softmax(dim=1)
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            yhat = sm(yhat)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc

    def train_evaluate_trvate(self, train_dl, valid_dl, test_dl, epochCnt=500):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        accvectr = np.zeros(epochCnt)
        accvecva = np.zeros(epochCnt)
        accvecte = np.zeros(epochCnt)
        for epoch in range(epochCnt):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
            acc_tr = self.evaluate_model(train_dl)
            acc_va = self.evaluate_model(valid_dl)
            acc_te = self.evaluate_model(test_dl)
            print("epoch ", epoch, "tr: %.3f" % acc_tr, "va: %.3f" % acc_va, "te: %.3f" % acc_te)
            accvectr[epoch] = acc_tr
            accvecva[epoch] = acc_va
            accvecte[epoch] = acc_te
        return accvectr, accvecva, accvecte

class MLP_Dict(Module):
    # define model elements
    def __init__(self, dim_of_input, dict_hidStates, classCount, dropout_value=None):

        super(MLP_Dict, self).__init__()

        # first fetch the hidden state variables as keys
        keys = [key for key, value in dict_hidStates.items()]
        print(keys)
        keysSorted = np.sort(keys)

        self.dropout_value = dropout_value
        self.dim_of_input = dim_of_input
        self.classCount = classCount
        self.keysSorted = keysSorted
        self.dict_hidStates = dict_hidStates
        self.initialize_net()

    def initialize_net(self):
        i = 0
        dim_in = self.dim_of_input
        for k in self.keysSorted:
            i = i + 1
            actFun = self.dict_hidStates[k]['act']
            dim_out = self.dict_hidStates[k]['dimOut']
            initMode = self.dict_hidStates[k]['initMode']
            print(i, k, initMode, dim_out, actFun)

            print("  self.{:s} = Linear({:d}, {:d})".format("hidden" + str(i), dim_in, dim_out))
            setattr(self, "hidden" + str(i), Linear(dim_in, dim_out))

            print("  kaiming_uniform_(self.{:s}.weight,nonlinearity={:s})".format("hidden" + str(i), actFun))
            kaiming_uniform_(getattr(self, "hidden" + str(i)).weight, nonlinearity=actFun)

            print("  self.{:s} = ReLU()".format("act" + str(i)))
            setattr(self, "act" + str(i), ReLU())

            if self.dropout_value is not None:
                print("  self.{:s} = Dropout({:2.1f})".format("dropout" + str(i), self.dropout_value))
                setattr(self, "dropout" + str(i), torch_do(p=self.dropout_value))

            dim_in = dim_out

        print("  self.finalLayer = Linear({:d},{:d})".format(dim_in, self.classCount))
        self.finalLayer = Linear(dim_in, self.classCount)
        xavier_uniform_(getattr(self, "hidden" + str(i)).weight)

    def forward(self, X):
        for i in range(len(self.keysSorted)):
            fhid = getattr(self, "hidden" + str(i+1))
            fact = getattr(self, "act" + str(i+1))
            X = fhid(X)
            X = fact(X)
        X = self.finalLayer(X)
        return X

    def load_model(self, model_file_full_path):
        print("loading parameters from : ", model_file_full_path)
        modelParams = torch_load(model_file_full_path)
        print("load_state_dict - ")
        self.load_state_dict(modelParams)
        self.eval()

    def train_model(self, train_dl, epochCnt=500):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.train()
        # enumerate epochs
        for epoch in range(epochCnt):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

    def export_final_layer(self, test_dl):
        final_layer, predictions, actuals = list(), list(), list()
        sm = Softmax(dim=1)
        self.eval()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.forward(inputs)

            fin_lay = yhat.clone()
            fin_lay = fin_lay.detach().numpy()
            final_layer.append(fin_lay)
            #if i<10:
            #   print(fin_lay.shape)

            yhat = sm(yhat)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        final_layer = np.vstack(final_layer)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc, predictions, actuals, final_layer

    def evaluate_model(self, test_dl):
        predictions, actuals = list(), list()
        sm = Softmax(dim=1)
        self.eval()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = self.forward(inputs)
            yhat = sm(yhat)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc, predictions, actuals

    def train_evaluate_trvate(self, train_dl, valid_dl, test_dl, epochCnt=500, saveBestModelName=None):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        accvectr = np.zeros(epochCnt)
        accvecva = np.zeros(epochCnt)
        accvecte = np.zeros(epochCnt)
        acc_va_max = 0

        for epoch in range(epochCnt):
            # enumerate mini batches
            self.train()
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = self.forward(inputs)
                # calculate loss
                loss = criterion(yhat, targets.squeeze_())
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()
            acc_tr, _, _ = self.evaluate_model(train_dl)
            acc_va, _, _ = self.evaluate_model(valid_dl)
            acc_te, preds_te, labels_te = self.evaluate_model(test_dl)

            if acc_va_max < acc_va:
                preds_best, labels_best = preds_te, labels_te
                print("best validation epoch so far - epoch ", epoch, "va: %.3f" % acc_va, "te: %.3f" % acc_te)
                acc_va_max = acc_va
                if saveBestModelName is not None:
                    print("Saving model at : ", saveBestModelName)
                    torch_save(self.state_dict(), saveBestModelName)
                    print("Model saved..")
            else:
                print("epoch ", epoch, "tr: %.3f" % acc_tr, "va: %.3f" % acc_va, "te: %.3f" % acc_te)

            accvectr[epoch] = acc_tr
            accvecva[epoch] = acc_va
            accvecte[epoch] = acc_te
        return accvectr, accvecva, accvecte, preds_best, labels_best