import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import importlib as impL

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

import vae_torch_model as vtm
import data_classes as dc
import vae_utils as vu

class HandCraftedDataset(Dataset):
    # load the dataset
    def __init__(self, path, X=None, y=None):
        # load the csv file as a dataframe
        if X is None and y is None:
            df = pd.read_csv(path, header=None)
            # store the inputs and outputs
            self.X = df.values[:, :-1]
            self.y = df.values[:, -1]
            # ensure input data is floats
            self.X = self.X.astype('float32')
            # label encode target and ensure the values are floats
            self.y = LabelEncoder().fit_transform(self.y)
            #print(self.X)
            #print(self.y)
        else:
            self.X = X.astype('float32')
            self.y = y
        print(self.X.shape)
        print(self.y.shape)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

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
            fhid = getattr(self, "hidden" + str(i + 1))
            fact = getattr(self, "act" + str(i + 1))
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
            # if i<10:
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

def load_data():
    X_tr = np.load('/home/doga/GithUBuntU/keyhandshapediscovery/vae_torch/tr_data.npz')
    X_va = np.load('/home/doga/GithUBuntU/keyhandshapediscovery/vae_torch/va_data.npz')
    X_te = np.load('/home/doga/GithUBuntU/keyhandshapediscovery/vae_torch/te_data.npz')
    print(X_tr.files)
    print("TrData=", X_tr['XTr'].shape, "TrMu=", X_tr['MuTr'].shape, "TrLab=", X_tr['labsTr'].shape)
    print("VaData=", X_va['XTr'].shape, "VaMu=", X_va['MuTr'].shape, "VaLab=", X_va['labsTr'].shape)
    print("TeData=", X_te['XTr'].shape, "TeMu=", X_te['MuTr'].shape, "TeLab=", X_te['labsTr'].shape)

    dataset_tr = HandCraftedDataset("", X=X_tr['MuTr'], y=X_tr['labsTr'])
    train_dl = DataLoader(dataset_tr, batch_size=32, shuffle=False)
    dataset_va = HandCraftedDataset("", X=X_va['MuTr'], y=X_va['labsTr'])
    valid_dl = DataLoader(dataset_va, batch_size=32, shuffle=False)
    dataset_te = HandCraftedDataset("", X=X_te['MuTr'], y=X_te['labsTr'])
    test_dl = DataLoader(dataset_te, batch_size=32, shuffle=False)

    X = {
        "trCnt": len(X_tr['labsTr']),
        "vaCnt": len(X_va['labsTr']),
        "teCnt": len(X_te['labsTr']),
        "ftCnt": X_tr['MuTr'].shape[1],
        "classCnt": len(np.unique(X_tr['labsTr'])),
        "train_dl": train_dl,
        "mTr": X_tr['MuTr'],
        "valid_dl": valid_dl,
        "mVa": X_va['MuTr'],
        "test_dl": test_dl,
        "mTe": X_te['MuTr'],
    }
    return X

def print_acc_from_X(X):
    if 'trLab' in X and 'trPrd' in X and len(X["trLab"])==len(X["trPrd"]):
        acc_tr = accuracy_score(X["trLab"], X["trPrd"])
        print("Initial training accuracy = ", acc_tr)
    if 'vaLab' in X and 'vaPrd' in X and len(X["vaLab"])==len(X["vaPrd"]):
        acc_va = accuracy_score(X["vaLab"], X["vaPrd"])
        print("Initial validation accuracy = ", acc_va)
    if 'teLab' in X and 'tePrd' in X and len(X["teLab"])==len(X["tePrd"]):
        acc_te = accuracy_score(X["teLab"], X["tePrd"])
        print("Initial test accuracy = ", acc_te)

def get_data_from_ConvVAE_model(data_main_fold="/media/doga/SSD258/DataPath/sup/data",
                                model_folder="/home/doga/GithUBuntU/keyhandshapediscovery/vae_torch/output_C18_is64_hs1296_fs64",
                                model_name="model_C18_is64_hs1296_fs64",
                                batch_size=32):
    save_data_name = "data_ConvVAE_" + model_name + ".npy"
    save_data_full_name = os.path.join(model_folder, save_data_name)
    if not os.path.exists(save_data_full_name):
        model = os.path.join(model_folder, model_name + ".model")
        input_size = 64
        #data_main_fold = "/media/doga/SSD258/DataPath/sup/data" #/home/doga/DataFolder/sup/data
        data_folder = os.path.join(data_main_fold, "data_XX_") #+ "/data_te2_cv1_neuralNetHandImages_nos11_rs224_rs01/neuralNetHandImages_nos11_rs224_rs01_XX_"
        X_tr = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=False, input_size=input_size)
        X_va = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
        X_te = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)

        mu_vec_tr, x_vec_tr, lab_vec_tr = vtm.ConvVAE.feat_extract_ext(model, X_tr, 64)
        mu_vec_va, x_vec_va, lab_vec_va = vtm.ConvVAE.feat_extract_ext(model, X_va, 64)
        mu_vec_te, x_vec_te, lab_vec_te = vtm.ConvVAE.feat_extract_ext(model, X_te, 64)

        X = {
            "trCnt": len(lab_vec_tr),
            "vaCnt": len(lab_vec_va),
            "teCnt": len(lab_vec_te),
            "ftCnt": mu_vec_tr.shape[1],
            "classCnt": len(np.unique(lab_vec_tr)),
            "train_dl": DataLoader(HandCraftedDataset("", X=mu_vec_tr, y=np.asarray(lab_vec_tr, dtype=int)), batch_size=batch_size, shuffle=False),
            "mTr": mu_vec_tr,
            "valid_dl": DataLoader(HandCraftedDataset("", X=mu_vec_va, y=np.asarray(lab_vec_va, dtype=int)), batch_size=batch_size, shuffle=False),
            "mVa": mu_vec_va,
            "test_dl" : DataLoader(HandCraftedDataset("", X=mu_vec_te, y=np.asarray(lab_vec_te, dtype=int)), batch_size=batch_size, shuffle=False),
            "mTe": mu_vec_te,
        }
        np.save(save_data_full_name, X, allow_pickle=True)
    else:
        X_L = np.load(save_data_full_name, allow_pickle=True)
        X = {}
        for k in X_L.item().keys():
            X[k] = X_L.item().get(k)

    return X

def get_data_from_ConvVAE_Multitask_model(data_main_fold="/home/doga/DataFolder/sup/data/conv_data",
                                model_folder="/home/doga/GithUBuntU/keyhandshapediscovery/output_ConvVAE_MultiTask_is64_hs9216_fs64",
                                model_name="model_ConvVAE_MultiTask_is64_hs9216_fs64",
                                batch_size=32):
    save_data_name = "data_ConvVAE_MultiTask_" + model_name + ".npy"
    save_data_full_name = os.path.join(model_folder, save_data_name)
    if not os.path.exists(save_data_full_name):
        model = os.path.join(model_folder, model_name + ".model")
        input_size = 64
        data_folder = os.path.join(data_main_fold, "data_XX_")
        X_tr = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=False, input_size=input_size)
        X_va = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
        X_te = dc.khs_dataset(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)

        mu_vec_tr, x_vec_tr, lab_vec_tr, pred_vec_tr = vtm.ConvVAE_MultiTask.feat_extract_ext(model, X_tr, 64)
        mu_vec_va, x_vec_va, lab_vec_va, pred_vec_va = vtm.ConvVAE_MultiTask.feat_extract_ext(model, X_va, 64)
        mu_vec_te, x_vec_te, lab_vec_te, pred_vec_te = vtm.ConvVAE_MultiTask.feat_extract_ext(model, X_te, 64)

        X = {
            "trCnt": len(lab_vec_tr),
            "vaCnt": len(lab_vec_va),
            "teCnt": len(lab_vec_te),
            "trLab": lab_vec_tr,
            "vaLab": lab_vec_va,
            "teLab": lab_vec_te,
            "trPrd": pred_vec_tr,
            "vaPrd": pred_vec_va,
            "tePrd": pred_vec_te,
            "ftCnt": mu_vec_tr.shape[1],
            "classCnt": len(np.unique(lab_vec_tr)),
            "train_dl": DataLoader(HandCraftedDataset("", X=mu_vec_tr, y=np.asarray(lab_vec_tr, dtype=int)), batch_size=batch_size, shuffle=False),
            "mTr": mu_vec_tr,
            "valid_dl": DataLoader(HandCraftedDataset("", X=mu_vec_va, y=np.asarray(lab_vec_va, dtype=int)), batch_size=batch_size, shuffle=False),
            "mVa": mu_vec_va,
            "test_dl" : DataLoader(HandCraftedDataset("", X=mu_vec_te, y=np.asarray(lab_vec_te, dtype=int)), batch_size=batch_size, shuffle=False),
            "mTe": mu_vec_te,
        }
        np.save(save_data_full_name, X, allow_pickle=True)
    else:
        X_L = np.load(save_data_full_name, allow_pickle=True)
        X = {}
        for k in X_L.item().keys():
            X[k] = X_L.item().get(k)

    print_acc_from_X(X)

    return X

def run_sup_learner(X=None, hidStateID=7, epochCnt=100,  applyPca=True):
    if X is None:
        X = load_data()
    hid_state_cnt_vec = vu.get_hid_state_vec(hidStateID)
    hidStatesDict = vu.create_hidstate_dict(hid_state_cnt_vec, init_mode_vec=None, act_vec=None)
    classCount = X["classCnt"]
    model_ = MLP_Dict(X["ftCnt"], hidStatesDict, classCount, dropout_value=0.3)

    print_acc_from_X(X)

    accvectr, accvecva, accvecte, preds_best, labels_best = model_.train_evaluate_trvate(X["train_dl"], X["valid_dl"],
                                                                                         X["test_dl"], epochCnt=epochCnt,
                                                                                         saveBestModelName=None)
    print("accvectr=", accvectr, ", accvecva=", accvecva, ", accvecte=", accvecte)
    return accvectr, accvecva, accvecte, preds_best, labels_best

"""
X = ss.get_data_from_ConvVAE_Multitask_model()
accvectr, accvecva, accvecte, preds_best, labels_best = ss.run_sup_learner(X=X)
"""

def run_compare_list(experiments_folder,
                     data_log_keys=['tr_te', 'te'],
                     loss_key_list=['bottleneck_act', 'bottleneck_kmeans', 'sparsity', 'reconstruction'],
                     exp_base_name='exp_conv_ae_simple_is28_cf',
                     ae_f_name_base='ae_ft_conv_ae_simple_is28.npy'):
    impL.reload(vu)
    cf_int_arr = np.array([454, 455, 458, 459, 460, 471, 472])
    title_add_front_str = 'kl-l2 best acc - '
    max_act_ep = None
    plot_average_win_size = 5
    save_to_fold = os.path.join(experiments_folder, 'kl_l2')
    vu.plot_cf_compare_list(cf_int_arr, data_log_keys, loss_key_list, title_add_front_str,
                            experiments_folder, exp_base_name, ae_f_name_base, max_act_ep=max_act_ep,
                            save_to_fold=save_to_fold, plot_average_win_size=plot_average_win_size)