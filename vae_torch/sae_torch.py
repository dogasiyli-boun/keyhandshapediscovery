import torch

import vae_torch_model
import vae_utils as vu
from data_classes import khs_dataset_v2
import vae_torch_model as vtm

import numpy as np
import datetime
import os
import helperFuncs as funcH
import pandas as pd

"""
import sys, importlib as impL
sys.path.insert(1,'vae_torch')
import vae_torch_model as vtm
import vae_torch as vt
from data_classes import khs_dataset_v2
import vae_scripts as ss

vt.main(epochs=100)
"""

def load_data(input_size, input_initial_resize, data_main_fold = "/home/doga/DataFolder/sup/data"):
    #"/media/doga/SSD258/DataPath/sup/data" #/home/doga/DataFolder/sup/data
    data_folder = os.path.join(data_main_fold, "data_XX_")
    X_tr_tr = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_tr"), is_train=True, input_size=input_size, input_initial_resize=input_initial_resize)
    X_tr_te = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_tr"), is_train=False, input_size=input_size)
    X_va = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
    X_te = khs_dataset_v2(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)
    return X_tr_tr, X_tr_te, X_va, X_te

def get_model(updatedModelFile=None,
              config_folder="/home/doga/GithUBuntU/keyhandshapediscovery/configs",
              config_file_id=0
              ):
    config_file = os.path.join(config_folder, "conf_autoencoder_"+str(config_file_id).zfill(2)+".yaml")
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device)
    print("Torch is in ", device, " mode.")
    if os.path.exists(updatedModelFile):
        _model_ae = torch.load(f=updatedModelFile)
        print("Model loaded from(", updatedModelFile, ")")
    elif os.path.exists(config_file):
        CONF_PARAMS_ = funcH.CustomConfigParser(config_file=config_file)
        _model_ae = vtm.Conv_AE_NestedNamespace(CONF_PARAMS_.MODEL).to(device)
        print("Model created from scratch.")

    return _model_ae

def main(model_name,
         epochs=20,
         data_main_fold="/media/doga/Data/VENVFOLD/vae_torch_data/conv_data_te2_va3_nos11",
         save_model_at_epochs=20,
         config_file_id=0):
    #data related variables
    input_initial_resize = 80
    input_size = 64
    X_tr_tr, X_tr_te, X_va, X_te = load_data(input_size, input_initial_resize, data_main_fold=data_main_fold)
    data_log_keys = ['tr_tr', 'tr_va', 'va', 'te']
    X_dict = {
        'tr_tr': X_tr_tr,
        'tr_va': X_tr_te,
        'va': X_va,
        'te': X_te,
    }

    batch_size = 64
    #experiment related variables
    base_str = model_name + "_is" + str(input_size)
    out_folder = 'output_' + base_str + '_cf' + str(config_file_id).zfill(2)
    ae_f_name = os.path.join(out_folder, 'ae_ft_' + base_str + '.npy')
    fig_loss_f_name = os.path.join(out_folder, 'XXX_fig_' + base_str + '.png')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    updatedModelFile = os.path.join(out_folder, "model_" + model_name + "_is" + str(input_size) + "_hs" + '.model')

    _model_ae = get_model(updatedModelFile=updatedModelFile)

    epoch_out_img_list = funcH.getFileList(out_folder, startString="output_va", endString=".png", sortList=False)
    ep_fr = len(epoch_out_img_list)

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
                loss_log_dict[k_data][epoch] = _model_ae.fit(X_dict[k_data], batch_size)
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
    main(model_name="sae_model",
         epochs=20,
         data_main_fold="/media/doga/Data/VENVFOLD/vae_torch_data/conv_data_te2_va3_nos11",
         save_model_at_epochs=20,
         config_file_id=0)
