import torch

import vae_torch_model
import vae_utils as vu
from data_classes import khs_dataset

import numpy as np
import datetime
import os

"""
import sys, importlib as impL
sys.path.insert(1,'vae_torch')
import vae_torch_model as vtm
import vae_torch as vt
from data_classes import khs_dataset
import vae_scripts as ss

vt.main(epochs=100)
"""

def getFileList(dir2Search, startString="", endString="", sortList=False):
    fileList = [f for f in os.listdir(dir2Search) if f.startswith(startString) and
                                                     f.endswith(endString) and
                                                    os.path.isfile(os.path.join(dir2Search, f))]
    if sortList:
        fileList = np.sort(fileList)
    return fileList

def load_data(input_size, input_initial_resize, data_main_fold = "/home/doga/DataFolder/sup/data"):
    #"/media/doga/SSD258/DataPath/sup/data" #/home/doga/DataFolder/sup/data
    data_folder = os.path.join(data_main_fold, "data_XX_")
    X_tr_tr = khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=True, input_size=input_size, input_initial_resize=input_initial_resize)
    X_tr_te = khs_dataset(root_dir=data_folder.replace("_XX_", "_tr"), is_train=False, input_size=input_size)
    X_va = khs_dataset(root_dir=data_folder.replace("_XX_", "_va"), is_train=False, input_size=input_size)
    X_te = khs_dataset(root_dir=data_folder.replace("_XX_", "_te"), is_train=False, input_size=input_size)
    return X_tr_tr, X_tr_te, X_va, X_te

def load_model_vars(model_name):
    if model_name == "ConvVAE_2":
        chn_sizes = [3, 32, 32, 16, 16, 16]
        kern_sizes = [5, 5, 5, 3, 3]
        hid_sizes = [16 * 9 * 9, 512]
    elif model_name == "ConvVAE" or model_name == "ConvVAE_MultiTask":
        chn_sizes = [3, 32, 32, 16]
        kern_sizes = [5, 5, 5]
        hid_sizes = [16 * 24 * 24, 512]
    return chn_sizes, kern_sizes, hid_sizes

def get_model(updatedModelFile, model_name, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size, apply_classification_task=False, class_count=None, update_weights=False):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(torch_device)
    print("Torch is in ", device, " mode.")
    if os.path.exists(updatedModelFile):
        _model_vae = torch.load(f=updatedModelFile)
    elif model_name == "ConvVAE":
        _model_vae = vae_torch_model.ConvVAE(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size).to(device)
    elif model_name == "ConvVAE_2":
        _model_vae = vae_torch_model.ConvVAE_2(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size).to(device)
    elif model_name == "ConvVAE_MultiTask":
        _model_vae = vae_torch_model.ConvVAE_MultiTask(input_size=input_size, chn_sizes=chn_sizes, kern_sizes=kern_sizes,
                                               hid_sizes=hid_sizes, feat_size=feat_size, class_count=class_count,
                                               apply_classification_task=apply_classification_task,
                                               update_weights_method=update_weights).to(device)

    return _model_vae

def main(model_name="ConvVAE_MultiTask",
         epochs=20, class_count=27, feat_size=64,
         data_main_fold="/media/doga/Data/VENVFOLD/vae_torch_data/conv_data_te2_va3_nos11",
         data_ident_str="conv_data_te2_va3_nos11",
         apply_classification_task=True, update_weights=True,
         save_figs_at_epochs=10,
         save_model_at_epochs=20):
    #data related variables
    input_initial_resize = 80
    input_size = 64
    X_tr_tr, X_tr_te, X_va, X_te = load_data(input_size, input_initial_resize, data_main_fold=data_main_fold)

    #learner related variables

    batch_size = 64
    #model related variables

    chn_sizes, kern_sizes, hid_sizes = load_model_vars(model_name)

    data_ident_str = "_" + data_ident_str if data_ident_str != "" else ""
    mt_str = "_mtTrue" if apply_classification_task else "_mtFalse"
    uw_str = "_uwTrue" if update_weights else "_uwFalse"

    #experiment related variables
    base_str = model_name + data_ident_str + "_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size) + mt_str + uw_str
    out_folder = 'output_' + base_str
    vae_f_name = os.path.join(out_folder, 'vae_ft_' + base_str + '.npz')
    fig_loss_f_name = os.path.join(out_folder, 'loss_fig_' + base_str + '.png')
    fig_acc_f_name = os.path.join(out_folder, 'acc_fig_' + base_str + '.png')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    updatedModelFile = os.path.join(out_folder, "model_" + model_name + "_is" + str(input_size) + "_hs" + str(hid_sizes[0]) + "_fs" + str(feat_size) + '.model')

    _model_vae = get_model(updatedModelFile, model_name, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size,
                           class_count=class_count, apply_classification_task=apply_classification_task, update_weights=update_weights)

    epoch_out_img_list = getFileList(out_folder, startString="output_val", endString=".png", sortList=False)
    ep_fr = len(epoch_out_img_list)

    if os.path.exists(vae_f_name):
        vfz = np.load(vae_f_name)
        tr_loss = list(vfz['tr_loss'])
        val_loss = list(vfz['val_loss'])
        tes_loss = list(vfz['tes_loss'])
        print("tr_loss is loaded with len", len(tr_loss))
        print("val_loss is loaded with len", len(val_loss))
        print("tes_loss is loaded with len", len(tes_loss))
        if vfz.files.__contains__('tr_acc'):
            tr_acc = list(vfz['tr_acc'])
        if vfz.files.__contains__('va_acc'):
            va_acc = list(vfz['va_acc'])
        if vfz.files.__contains__('te_acc'):
            te_acc = list(vfz['te_acc'])
        if vfz.files.__contains__('sumBCE'):
            sumBCE = list(vfz['sumBCE'])
        if vfz.files.__contains__('sumCLS'):
            sumCLS = list(vfz['sumCLS'])
        if vfz.files.__contains__('sumKLD'):
            sumKLD = list(vfz['sumKLD'])
        if vfz.files.__contains__('wBCE'):
            wBCE = list(vfz['wBCE'])
        if vfz.files.__contains__('wCLS'):
            wCLS = list(vfz['wCLS'])
        if vfz.files.__contains__('wKLD'):
            wKLD = list(vfz['wKLD'])

    else:
        tr_loss = []
        val_loss = []
        tes_loss = []
        tr_acc = []
        va_acc = []
        te_acc = []
        sumBCE = []
        sumCLS = []
        sumKLD = []
        wBCE = []
        wCLS = []
        wKLD = []

    for epoch in range(ep_fr, epochs):
        print(f"Epoch {epoch} of {epochs}")

        loss_tr_tr = _model_vae.fit(X_tr_tr, batch_size)
        loss_te_tr = _model_vae.validate(X_tr_te, epoch, batch_size, out_folder, out_name_add_str="tra_")
        loss_te_va = _model_vae.validate(X_va, epoch, batch_size, out_folder, out_name_add_str="val_")
        loss_te_te = _model_vae.validate(X_te, epoch, batch_size, out_folder, out_name_add_str="tes_")

        sumBCE.append(loss_tr_tr["sumBCE"])
        sumCLS.append(loss_tr_tr["sumCLS"])
        sumKLD.append(loss_tr_tr["sumKLD"])
        wBCE.append(loss_tr_tr["wBCE"])
        wCLS.append(loss_tr_tr["wCLS"])
        wKLD.append(loss_tr_tr["wKLD"])
        tr_loss.append(loss_te_tr["loss"])
        val_loss.append(loss_te_va["loss"])
        tes_loss.append(loss_te_te["loss"])
        print(f"Train Loss: {loss_te_tr['loss']:.4f}, Val Loss: {loss_te_va['loss']:.4f}, Tes Loss: {loss_te_te['loss']:.4f}")
        print("tr last 5", tr_loss[-5:])
        print("val last 5", val_loss[-5:])
        print("tes last 5", tes_loss[-5:])

        if model_name.__contains__("MultiTask"):
            tr_acc.append(loss_te_tr["acc"])
            va_acc.append(loss_te_va["acc"])
            te_acc.append(loss_te_te["acc"])
            print(f"tr_acc: {loss_te_tr['acc']:.2f}, va_acc: {loss_te_va['acc']:.2f}, acc_te: {loss_te_te['acc']:.2f}")
            print("trLast 5", tr_acc[-5:])
            print("vaLast 5", va_acc[-5:])
            print("teLast 5", te_acc[-5:])

        print(datetime.datetime.now().strftime("%H:%M:%S"))

        torch.save(_model_vae, f=updatedModelFile)
        if (epoch>0 and epoch%save_model_at_epochs==0):
            torch.save(_model_vae, f=str(updatedModelFile).replace(".model", "_ep{:03d}.model".format(epoch)))

        np.savez(vae_f_name,
                 tr_loss=tr_loss, val_loss=val_loss, tes_loss=tes_loss,
                 tr_acc=tr_acc, va_acc=va_acc, te_acc=te_acc,
                 sumBCE=sumBCE, sumKLD=sumKLD, sumCLS=sumCLS,
                 wBCE = wBCE, wKLD = wKLD, wCLS = wCLS)

        if (epoch>0 and epoch%save_figs_at_epochs==0):
            plt_x = {'tr_loss': tr_loss, 'val_loss': val_loss, 'tes_loss': tes_loss}
            vu.plot_vars(X=plt_x, contain_str='_loss',  ylabel="Loss", title_str="Loss on " + base_str, save_file_name=fig_loss_f_name)
            plt_x = {'tr_acc': tr_acc, 'va_acc': va_acc, 'te_acc': te_acc}
            vu.plot_vars(X=plt_x, contain_str='_acc',  ylabel="Accuracy", title_str="Accuracy on " + base_str, save_file_name=fig_acc_f_name)


if __name__ == '__main__':
    main(model_name="ConvVAE_MultiTask",
         epochs=20, class_count=27, feat_size=64,
         data_main_fold="/home/doga/DataFolder/sup/data/conv_data",
         data_ident_str="conv_data_te2_va3_nos11",
         apply_classification_task=True, update_weights=True,
         save_figs_at_epochs=10, save_model_at_epochs=20)
