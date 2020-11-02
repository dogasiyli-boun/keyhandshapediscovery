import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Sparsity_Loss_Base(nn.Module):
    def __init__(self):
        super(Sparsity_Loss_Base, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sparse_KL_DivergenceLoss(Sparsity_Loss_Base):
    def __init__(self, rho=None, rho_one_mode=False, rho_one_mode_perc=None,
                 reduction='batchmean',
                 apply_log_soft_max=True, apply_sigmoid=False, apply_mean=False):
        super(Sparse_KL_DivergenceLoss, self).__init__()
        self.rho = rho
        self.rho_set = False
        self.reduction = reduction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.apply_sigmoid = apply_sigmoid
        self.apply_log_soft_max = apply_log_soft_max
        self.apply_mean = apply_mean
        self.rho_one_mode = rho_one_mode
        self.kl_rho_one_mode_perc = rho_one_mode_perc
        self.kl_rho_one_vec = None
        print("kl_divergence.apply_sigmoid = ", self.apply_sigmoid)
        print("kl_divergence.apply_log_soft_max = ", self.apply_log_soft_max)
        print("kl_divergence.apply_mean = ", self.apply_mean)
        print("kl_divergence.rho_one_mode = ", self.rho_one_mode)
        print("kl_rho_one_mode_perc = ", self.kl_rho_one_mode_perc)

    def set_kl_rho_by_data(self, bt): #  bt: bottleneck
        if self.rho_set:
            return
        if self.rho_one_mode:
            self.rho_set = True
            print("self.kl_rho will be skipped because rho_one=True")
            if self.kl_rho_one_mode_perc is None:
                self.kl_rho_one_mode_perc = [0.25, 0.10]
            else:
                #  self.kl_rho_one_mode_perc = '0.25/0.10'
                vec_fr_str = np.asarray(str(self.kl_rho_one_mode_perc).split('/'))
                self.kl_rho_one_vec = vec_fr_str.astype(np.float).squeeze()
            return
        bt_dim = bt.shape[1]
        wanted_rho = self.rho
        self.rho = 1 / bt_dim
        if self.rho is None:
            print("self.kl_rho is set to {:f}, dimension of bottleneck is {:d}".format(1 / bt_dim, bt_dim))
        elif wanted_rho != 1/bt_dim:
            print("self.kl_rho changed from {:f} to {:f}, dimension of bottleneck is {:d}".format(wanted_rho, 1/bt_dim, bt_dim))

        self.rho_set = True

    def find_predicted_clusters(self, bt):
        bt_copy = bt.to(torch.device('cpu')).detach().numpy()
        N, D = bt_copy.shape
        if self.kl_rho_one_vec is None:
            predicted_clusters_decided = np.argmax(bt_copy, axis=1).squeeze()
        else:
            count = len(self.kl_rho_one_vec)+1
            predicted_clusters = np.zeros((count, N), dtype=int)
            max_vals = np.zeros((count, N), dtype=float)
            for i in range(count-1):
                predicted_clusters[i, :] = np.argmax(bt_copy, axis=1).squeeze()
                max_vals[i, :] = bt_copy[np.array(range(0, N)), predicted_clusters[i, :]]
                bt_copy[np.array(range(0, N)), predicted_clusters[i, :]] = 0
            predicted_clusters[count-1, :] = np.random.randint(low=0, high=D, size=N, dtype=int)

            predicted_clusters_decided = np.zeros((1, N), dtype=int).squeeze() - 1
            rand_sample_ids = np.array(np.random.permutation(np.arange(N)), dtype=int)
            kl_rho_one_vec = np.cumsum(np.concatenate([np.array([1-np.sum(self.kl_rho_one_vec)], float), self.kl_rho_one_vec]))
            fr = int(0)
            for i in range(count):
                to = int(np.floor(kl_rho_one_vec[i]*N))
                predicted_clusters_decided[rand_sample_ids[fr:to]] = predicted_clusters[i, rand_sample_ids[fr:to]]
                fr = to
        return predicted_clusters_decided

    def forward(self, bt):
        # https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/13
        # KLDLoss(p, q), sum(q) needs to equal one
        # p = log_softmax(tensor)
        self.set_kl_rho_by_data(bt)
        if self.apply_sigmoid:
            bt = torch.sigmoid(bt)
        if not self.rho_one_mode and self.apply_mean:
            #"if in rho_one_mode can not apply mean"
            bt = torch.mean(bt, 1)
        if self.rho_one_mode:
            rho_mat = torch.zeros(bt.size(), dtype=torch.float32).to(self.device)
            predicted_clusters_decided = self.find_predicted_clusters(bt)
            # print(predicted_clusters_decided)
            #rho_val = float(torch.mean(torch.mean(bt)) / 4)
            #bt_add = (torch.randint(2, bt.size()) * torch.tensor([rho_val] * np.ones(bt.size()), dtype=torch.float32)).to(self.device)
            rho_mat[range(bt.size(0)), predicted_clusters_decided] = 1
        else:
            rho_mat = torch.tensor([self.rho] * np.ones(bt.size()), dtype=torch.float32).to(self.device)
        # rho_mat = torch.tensor([self.rho] * len(bt)).to(self.device)
        if self.apply_log_soft_max:
            loss_ret_1 = F.kl_div(F.log_softmax(bt, dim=1).to(self.device), rho_mat, reduction=self.reduction)
        else:
            loss_ret_1 = F.kl_div(bt, rho_mat, reduction=self.reduction)
        return loss_ret_1

class Sparse_Loss_Dim(Sparsity_Loss_Base):
    def __init__(self, dim, reduction='batchmean'):
        super(Sparse_Loss_Dim, self).__init__()
        self.dim = dim
        self.reduction = reduction

    # https: // discuss.pytorch.org / t / how - torch - norm - works - and -how - it - calculates - l1 - and -l2 - loss / 58387

    @staticmethod
    def l2_norm(bt, reduction):
        loss_ret_1 = torch.sum(((bt * bt)) ** 2, 0).sqrt()
        # loss_ret_2 = torch.norm(((bt.transpose() * bt.transpose())), 2, -1)
        # loss_ret_3 = torch.mean(torch.pow(bt, 2.0)).sqrt()
        if reduction == 'batchmean':
            #bunu kullanınca hepsi birbirine eşitleniyo
            loss_ret_1 = torch.mean(loss_ret_1)
        else:
            #bunu kullanınca tek bir node bütün sample'larda active oluyor
            loss_ret_1 = torch.sum(loss_ret_1)
        return loss_ret_1

    @staticmethod
    def l1_norm(bt, reduction):
        loss_ret_1 = torch.sum(torch.abs(bt), 0)
        # loss_ret_2 = torch.norm(((bt * bt)), 1, -1)
        if reduction == 'batchmean':
            loss_ret_1 = torch.mean(loss_ret_1)
        else:
            loss_ret_1 = torch.sum(loss_ret_1)
        return loss_ret_1

    def forward(self, bt):
        if self.dim == 1:
            return self.l1_norm(bt, self.reduction)
        if self.dim == 2:
            return self.l2_norm(bt, self.reduction)
        os.error("unknown dimension")

class Sparse_Loss_CrossEntropy(Sparsity_Loss_Base):
    def __init__(self, reduction='mean', apply_sigmoid_activation=False):
        super(Sparse_Loss_CrossEntropy, self).__init__()
        self.sigmoidAct = apply_sigmoid_activation
        self.loss_fun = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, bt):
        if self.sigmoidAct:
            bt = torch.sigmoid(bt)  # sigmoid because we need the probability distributions
        _, preds = torch.max(bt, 1)
        loss_ret_1 = self.loss_fun(bt, preds)
        return loss_ret_1