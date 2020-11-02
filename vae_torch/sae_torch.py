import torch

from data_classes import khs_dataset_v2, fashion_mnist, cifar10
import vae_torch_model as vtm

import numpy as np
import datetime
import os
import helperFuncs as funcH

"""
import sys, importlib as impL
sys.path.insert(1,'vae_torch')
import vae_torch_model as vtm
import vae_torch as vt
from data_classes import khs_dataset_v2
import vae_scripts as ss

vt.main(epochs=100)
"""

def load_hospisign_data(input_size, input_initial_resize, data_main_fold = "/home/doga/DataFolder/sup/data"):
    #"/media/doga/SSD258/DataPath/sup/data" #/home/doga/DataFolder/sup/data
    data_folder = os.path.join(data_main_fold, "data_XX_")
    X_tr_tr = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_tr"), is_train=True, input_size=input_size, input_initial_resize=input_initial_resize)
    X_tr_te = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_tr"), is_train=False, input_size=input_size)
    X_va = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
    X_te = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)
    return X_tr_tr, X_tr_te, X_va, X_te

def get_model(updatedModelFile=None, CONF_PARAMS_=None):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device)
    print("Torch is in ", device, " mode.")
    if os.path.exists(updatedModelFile):
        _model_ae = torch.load(f=updatedModelFile)
        print("Model loaded from(", updatedModelFile, ")")
        _model_ae.update_old_models()
    elif CONF_PARAMS_ is not None:
        _model_ae = vtm.Conv_AE_NestedNamespace(CONF_PARAMS_.MODEL).to(device)
        print("Model created from scratch.")
    return _model_ae

def get_data(CONF_PARAMS_):
    # data related variables
    data_main_fold = str(CONF_PARAMS_.DIR.DATA)
    if CONF_PARAMS_.DATA.IDENTIFIER == 'HOSPISIGN':
        input_initial_resize = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.DATA, 'INPUT_INITIAL_RESIZE', default_type=int, default_val=80)
        input_size = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.MODEL, 'INPUT_SIZE', default_type=int, default_val=64)
        X_tr_tr, X_tr_te, X_va, X_te = load_hospisign_data(input_size, input_initial_resize, data_main_fold=data_main_fold)
        data_log_keys = ['tr_tr', 'tr_va', 'va', 'te']
        X_dict = {
            'tr_tr': X_tr_tr,
            'tr_va': X_tr_te,
            'va': X_va,
            'te': X_te,
        }
    elif CONF_PARAMS_.DATA.IDENTIFIER == 'FASHION_MNIST':
        input_initial_resize = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.DATA, 'INPUT_INITIAL_RESIZE',
                                                                         default_type=int, default_val=34)
        load_train_as_test = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.DATA, 'LOAD_TR_AS_TEST',
                                                                       default_type=bool, default_val=False)
        input_size = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.MODEL, 'INPUT_SIZE', default_type=int,
                                                               default_val=28)
        X_tr_tr = fashion_mnist(fashionMNISTds_fold=data_main_fold, is_train=True, input_size=input_size,
                                load_train_as_test=load_train_as_test, input_initial_resize=input_initial_resize,
                                datasetname="fashion_mnist_tr_tr")
        X_tr_te = fashion_mnist(fashionMNISTds_fold=data_main_fold, is_train=True, input_size=input_size,
                                input_initial_resize=None, load_train_as_test=True, datasetname="fashion_mnist_tr_te")
        X_te = fashion_mnist(fashionMNISTds_fold=data_main_fold, is_train=False, input_size=input_size,
                             input_initial_resize=None, datasetname="fashion_mnist_te")
        data_log_keys = ['tr_tr', 'tr_te', 'te']
        X_dict = {
            'tr_tr': X_tr_tr,
            'tr_te': X_tr_te,
            'te': X_te,
        }
    elif CONF_PARAMS_.DATA.IDENTIFIER == 'CIFAR10':
        input_initial_resize = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.DATA, 'INPUT_INITIAL_RESIZE',
                                                                         default_type=int, default_val=34)
        load_train_as_test = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.DATA, 'LOAD_TR_AS_TEST',
                                                                       default_type=bool, default_val=False)
        input_size = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.MODEL, 'INPUT_SIZE', default_type=int,
                                                               default_val=28)
        X_tr_tr = cifar10(cifar10ds_fold=data_main_fold, is_train=True, input_size=input_size,
                          load_train_as_test=load_train_as_test, input_initial_resize=input_initial_resize,
                          datasetname="cifar10_tr_tr")
        X_tr_te = cifar10(cifar10ds_fold=data_main_fold, is_train=True, input_size=input_size,
                          input_initial_resize=None, load_train_as_test=True, datasetname="cifar10_tr_te")
        X_te = cifar10(cifar10ds_fold=data_main_fold, is_train=False, input_size=input_size,
                       input_initial_resize=None, datasetname="cifar10_te")
        data_log_keys = ['tr_tr', 'tr_te', 'te']
        X_dict = {
            'tr_tr': X_tr_tr,
            'tr_te': X_tr_te,
            'te': X_te,
        }
    else:
        os.error(22)
    return X_dict, data_log_keys, input_size

def get_last_epoch_completed(out_folder):
    epoch_out_img_list = funcH.getFileList(out_folder, startString="output_te", endString=".png", sortList=False)
    ep_fr = len(epoch_out_img_list)
    return ep_fr

def main(epochs=20, config_folder="/home/doga/GithUBuntU/keyhandshapediscovery/configs", config_file_id=0):
    config_file = os.path.join(config_folder, "conf_autoencoder_" + str(config_file_id).zfill(2) + ".yaml")
    CONF_PARAMS_ = funcH.CustomConfigParser(config_file=config_file)

    model_name = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.MODEL, 'MODEL_NAME', default_type=str, default_val='conv_ae_model')
    experiment_main_fold = str(CONF_PARAMS_.DIR.EXPERIMENT)
    save_model_at_epochs = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.EXPERIMENT, 'SAVE_MODEL_AT', default_type=int, default_val=20)
    batch_size = funcH.get_attribute_from_nested_namespace(CONF_PARAMS_.EXPERIMENT, 'BATCH_SIZE', default_type=int, default_val=64)

    X_dict, data_log_keys, input_size = get_data(CONF_PARAMS_)

    #experiment related variables
    base_str = model_name + "_is" + str(input_size)
    out_folder = os.path.join(experiment_main_fold, 'exp_' + base_str + '_cf' + str(config_file_id).zfill(2))
    ae_f_name = os.path.join(out_folder, 'ae_ft_' + base_str + '.npy')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    updatedModelFile = os.path.join(out_folder, "model_" + model_name + "_is" + str(input_size) + "_hs" + '.model')
    _model_ae = get_model(updatedModelFile=updatedModelFile, CONF_PARAMS_=CONF_PARAMS_)
    ep_fr = get_last_epoch_completed(out_folder)

    if os.path.exists(ae_f_name):
        vfz = np.load(ae_f_name, allow_pickle=True)
        loss_log_dict = {}
        for k in data_log_keys:
            loss_log_dict[k] = vfz.item().get(k)
            print(k, " - log is loaded with len: ", len(loss_log_dict[k]))
        loss_key_list = [key for key in loss_log_dict[data_log_keys[0]][0].keys()]
    else:
        loss_log_dict = {}
        for k in data_log_keys:
            loss_log_dict[k] = {}
        loss_key_list = None

    for epoch in range(ep_fr, epochs):
        print("*-*-*-*-*-*-*-*")
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        print(f"Epoch {epoch} of {epochs}")

        for k_data in data_log_keys:
            if k_data == 'tr_tr':
                print("training:", k_data, '--', datetime.datetime.now().strftime("%H:%M:%S"))
                loss_log_dict[k_data][epoch] = _model_ae.fit(X_dict[k_data], batch_size, out_folder=out_folder, epoch=epoch)
            else:
                print("evaluating:", k_data, '--', datetime.datetime.now().strftime("%H:%M:%S"))
                loss_log_dict[k_data][epoch] = _model_ae.validate(X_dict[k_data], epoch, batch_size, out_folder, sub_data_identifier=k_data + "_")

        if loss_key_list is None:
            loss_key_list = [key for key in loss_log_dict["tr_tr"][epoch].keys()]

        #for k in loss_key_list:
        for k_loss in loss_key_list:
            print(k_loss, ':')
            for k_data in data_log_keys:
                if k_loss in loss_log_dict[k_data][0]:
                    los_vec_cur = [loss_log_dict[k_data][l][k_loss] for l in range(0, len(loss_log_dict[k_data]))]
                    print('--', k_data, '-->', los_vec_cur[-5:])

        torch.save(_model_ae, f=updatedModelFile)
        if (epoch>0 and epoch%save_model_at_epochs==0):
            torch.save(_model_ae, f=str(updatedModelFile).replace(".model", "_ep{:03d}.model".format(epoch)))

        np.save(ae_f_name, loss_log_dict, allow_pickle=True)

if __name__ == '__main__':
    main(epochs=20,
         config_folder="/home/doga/GithUBuntU/keyhandshapediscovery/configs",
         config_file_id=0)
