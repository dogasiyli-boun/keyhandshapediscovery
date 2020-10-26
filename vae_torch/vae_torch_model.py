#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from numpy import unique as np_unique
from torch.utils.data import DataLoader
import numpy as np
from sys import exit as sys_exit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

import helperFuncs as funcH

class Flatten(nn.Module):
    '''
    This model is for conv autoencoders
    At some point the convolution layers turn to Linear layers
    This layer will be used for flattening and unflattening such layers
    '''
    def __init__(self):
        super(Flatten, self).__init__()
        self.in_size = None

    def forward(self, input):
        if self.in_size is None:
            self.in_size = [input.size(1), input.size(2), input.size(3)]
        return input.view(input.size(0), -1)

    def backward(self, input):
        return input.view(input.size(0), self.in_size[0], self.in_size[1], self.in_size[2])

    def flatten(self, input):
        return self.forward(input)

    def unflatten(self, input):
        return self.backward(input)

# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self, input_size, hid_sizes, feat_size):
        super(LinearVAE, self).__init__()

        self.in_linear_size = input_size*input_size
        self.input_size = input_size
        self.hid_sizes = hid_sizes
        self.feat_size = feat_size

        # encoder
        self.enc1 = nn.Linear(in_features=self.in_linear_size, out_features=hid_sizes[0])
        self.enc2 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.enc3 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[2])
        self.enc4 = nn.Linear(in_features=hid_sizes[2], out_features=feat_size*2)

        # decoder
        self.dec1 = nn.Linear(in_features=feat_size, out_features=hid_sizes[2])
        self.dec2 = nn.Linear(in_features=hid_sizes[2], out_features=hid_sizes[1])
        self.dec3 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.dec4 = nn.Linear(in_features=hid_sizes[0], out_features=self.in_linear_size)

        print("input_size=", input_size)
        print("in_linear_size=", self.in_linear_size)
        print("hid_sizes=", hid_sizes)
        print("feat_size=", feat_size)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = self.enc4(x).view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        reconstruction = torch.sigmoid(self.dec4(x))
        return reconstruction, mu, log_var

# define a Convolutional VAE
class ConvVAE(nn.Module):
    def __init__(self, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size):
        super(ConvVAE, self).__init__()

        self.input_size = input_size  # 64
        self.chn_sizes = chn_sizes  # chn_sizes = [3, 32, 32, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5, 5, 5]
        self.hid_sizes = hid_sizes  # hid_sizes = [9216(16*24*24), 512]
        self.feat_size = feat_size  # feat_size = 64
        self.bottle_neck_image_size = None

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L5_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L6_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L8_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L9_upsm1 = nn.Upsample(scale_factor=2)
        self.L10_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L11_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        if self.bottle_neck_image_size is None:
            self.bottle_neck_image_size = [x.size(1), x.size(2), x.size(3)]
        x = x.view(x.size(0), -1)
        x = F.relu(self.L4_lenc1(x))
        x = self.L5_lenc2(x)
        return x

    def dec(self, z):
        # decoding
        x = F.relu(self.L6_ldec2(z))
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), self.bottle_neck_image_size[0], self.bottle_neck_image_size[1], self.bottle_neck_image_size[2])
        x = F.relu(self.L8_dcnv3(x))
        x = self.L9_upsm1(x)
        x = F.relu(self.L10_dcnv2(x))
        reconstruction = torch.sigmoid(self.L11_dcnv1(x))
        return reconstruction

    def forward(self, x):
        x = self.enc(x)

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        reconstruction = self.dec(z)

        return reconstruction, mu, log_var

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def fit(self, X_data, batch_size):
        self.train()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        #lab_vec = []
        for b in dloader:
            data = b['image']
            #labels = b['label']
            #lab_vec.append(labels)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.forward(data)
            bce_loss = self.criterion(reconstruction, data)
            loss = self.final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss/len(X_data)
        return train_loss

    def validate(self, X_vate, epoch, batch_size, out_folder, out_name_add_str=""):
        self.eval()
        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)

        with torch.no_grad():
            fr = 0
            while (fr + batch_size < len(X_vate)):
                to = fr + batch_size
                data = torch.stack(batch[fr:to], dim=0)
                data = data.to(self.device)
                reconstruction, mu, logvar = self.forward(data)
                bce_loss = self.criterion(reconstruction, data)
                loss = self.final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                fr = to

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_" + out_name_add_str + "{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)
        val_loss = running_loss/len(X_vate)
        return val_loss

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L4_lenc1(x))
                x = model.L5_lenc2(x)
                x_vec.append(x)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=batch_lb)
        return mu_vec, x_vec, lab_vec

    def feat_extract(self, X_vate, batch_size):
        return self.feat_extract_ext(self, X_vate, batch_size)

class ConvVAE_2(nn.Module):
    def __init__(self, input_size, chn_sizes, kern_sizes, hid_sizes, feat_size, droput_val=None):
        super(ConvVAE_2, self).__init__()

        self.input_size = input_size  # 64            0   1   2   3   4   5
        self.chn_sizes = chn_sizes  # chn_sizes =    [3, 32, 32, 16, 16, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5,  5,  5,  3,  3]
        self.hid_sizes = hid_sizes  # hid_sizes =    [1600, 256]
        self.feat_size = feat_size  # feat_size = 64

        self.DropLayer = nn.Dropout(p=droput_val) if droput_val is not None else None

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_conv4 = nn.Conv2d(in_channels=chn_sizes[3], out_channels=chn_sizes[4], kernel_size=kern_sizes[3], stride=1, padding=0)
        self.L5_maxp2 = nn.MaxPool2d(kernel_size=2)
        self.L6_conv5 = nn.Conv2d(in_channels=chn_sizes[4], out_channels=chn_sizes[5], kernel_size=kern_sizes[4], stride=1, padding=0)
        self.L7_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L8_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L8_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L6_dcnv5 = nn.ConvTranspose2d(in_channels=chn_sizes[5], out_channels=chn_sizes[4], kernel_size=kern_sizes[4], stride=1, padding=0)
        self.L5_upsm1 = nn.Upsample(scale_factor=2)
        self.L4_dcnv4 = nn.ConvTranspose2d(in_channels=chn_sizes[4], out_channels=chn_sizes[3], kernel_size=kern_sizes[3], stride=1, padding=0)
        self.L3_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L2_upsm1 = nn.Upsample(scale_factor=2)
        self.L1_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L0_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss(reduction='sum')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        x = F.relu(self.L4_conv4(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L5_maxp2(x)
        x = F.relu(self.L6_conv5(x))
        rp0, rp1, rp2 = x.size(1), x.size(2), x.size(3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.L7_lenc1(x))
        if self.DropLayer is not None:
            x = self.DropLayer(x)
        x = self.L8_lenc2(x)
        return x, rp0, rp1, rp2

    def dec(self, x, rp0, rp1, rp2):
        # decoding
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), rp0, rp1, rp2)
        x = F.relu(self.L6_dcnv5(x))
        x = self.L5_upsm1(x)
        x = F.relu(self.L4_dcnv4(x))
        x = F.relu(self.L3_dcnv3(x))
        x = self.L2_upsm1(x)
        x = F.relu(self.L1_dcnv2(x))
        reconstruction = torch.sigmoid(self.L0_dcnv1(x))
        return reconstruction

    def forward(self, x):
        x, rp0, rp1, rp2 = self.enc(x)

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        x = F.relu(self.L8_ldec2(z))

        reconstruction = self.dec(x, rp0, rp1, rp2)

        return reconstruction, mu, log_var

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def fit(self, X_data, batch_size):
        self.train()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        #lab_vec = []
        for b in dloader:
            data = b['image']
            #labels = b['label']
            #lab_vec.append(labels)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar = self.forward(data)
            bce_loss = self.criterion(reconstruction, data)
            loss = self.final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss = running_loss/len(X_data)
        return train_loss

    def validate(self, X_vate, epoch, batch_size, out_folder):
        self.eval()
        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)

        with torch.no_grad():
            fr = 0
            while (fr + batch_size < len(X_vate)):
                to = fr + batch_size
                data = torch.stack(batch[fr:to], dim=0)
                data = data.to(self.device)
                reconstruction, mu, logvar = self.forward(data)
                bce_loss = self.criterion(reconstruction, data)
                loss = self.final_loss(bce_loss, mu, logvar)
                running_loss += loss.item()
                fr = to

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            #data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)
        val_loss = running_loss/len(X_vate)
        return val_loss

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = F.relu(model.L4_conv4(x))
                x = model.L5_maxp2(x)
                x = F.relu(model.L6_conv5(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L7_lenc1(x))
                x = model.L8_lenc2(x)

                x_vec.append(x)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=batch_lb)
        return mu_vec, x_vec, lab_vec

def get_torch_layer_from_dict(definiton_dict):
    if definiton_dict["type"] == 'Conv2d':
        #{"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size,"stride":stride,"padding":padding}
        in_channels = definiton_dict["in_channels"]
        out_channels = definiton_dict["out_channels"]
        kernel_size = definiton_dict["kernel_size"]
        stride = funcH.get_attribute_from_dict(definiton_dict, "stride", default_type=int, default_val=1)
        padding = funcH.get_attribute_from_dict(definiton_dict, "padding", default_type=int, default_val=0)
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding)

    if definiton_dict["type"] == 'ConvTranspose2d':
        #{"in_channels":in_channels,"out_channels":out_channels,"kernel_size":kernel_size,"stride":stride,"padding":padding}
        in_channels = definiton_dict["in_channels"]
        out_channels = definiton_dict["out_channels"]
        kernel_size = definiton_dict["kernel_size"]
        stride = funcH.get_attribute_from_dict(definiton_dict, "stride", default_type=int, default_val=1)
        padding = funcH.get_attribute_from_dict(definiton_dict, "padding", default_type=int, default_val=0)
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding)

    if definiton_dict["type"] == 'MaxPool2d':
        kernel_size = funcH.get_attribute_from_dict(definiton_dict, "kernel_size", default_type=int, default_val=2)
        return nn.MaxPool2d(kernel_size=kernel_size)

    if definiton_dict["type"] == 'Upsample':
        scale_factor = funcH.get_attribute_from_dict(definiton_dict, "scale_factor", default_type=int, default_val=2)
        return nn.Upsample(scale_factor=scale_factor)

    if definiton_dict["type"] == 'Linear':
        #{"in_features":in_features,"out_channels":out_features}
        in_features = definiton_dict["in_features"]
        out_features = definiton_dict["out_features"]
        return nn.Linear(in_features=in_features, out_features=out_features)

    if definiton_dict["type"] == 'ReLu':
        return nn.ReLU()

    if definiton_dict["type"] == 'Softmax':
        dim = funcH.get_attribute_from_dict(definiton_dict, "dim", default_type=int, default_val=0)
        return nn.Softmax(dim=dim)

    if definiton_dict["type"] == 'Flatten':
        return Flatten()

    if definiton_dict["type"] == 'Unflatten':
        return None

    sys_exit("Unknown type" + definiton_dict["type"])

def params_string_to_dict(params_string):
    params_dict = {}
    for s in params_string.split(','):
        k, v = s.split(':')
        try:
            params_dict[k.replace(" ", "")] = int(v)
        except:
            params_dict[k.replace(" ", "")] = v.replace(" ", "")
    return params_dict

def get_encoder_from_ns(x, verbose=0):
    layer_list = {}
    for k in vars(x):
        _att = getattr(x, k)
        layer_dict = params_string_to_dict(_att)
        if verbose > 0:
            print(k, layer_dict)
        layer_2_Add = get_torch_layer_from_dict(layer_dict)
        layer_list[k] = {'layer_type': layer_dict['type'], 'layer_module': layer_2_Add}
    return layer_list

def get_decoder_from_ns(x, verbose=0):
    layer_list = {}
    for k in vars(x):
        _att = getattr(x, k)
        layer_dict = params_string_to_dict(_att)
        if verbose > 0:
            print(k, layer_dict)
        if layer_dict['type'] == 'Unflatten':
            layer_2_Add = None
        else:
            layer_2_Add = get_torch_layer_from_dict(layer_dict)
        layer_list[k] = {'layer_type': layer_dict['type'], 'layer_module': layer_2_Add}
    return layer_list

class ConvVAE_MultiTask(nn.Module):
    def __init__(self,
                 input_size,
                 chn_sizes, kern_sizes, hid_sizes, feat_size,
                 class_count, apply_classification_task=True,
                 random_seed= 0,
                 update_weights_method=None):
        super(ConvVAE_MultiTask, self).__init__()

        self.input_size = input_size  # 64
        self.chn_sizes = chn_sizes  # chn_sizes = [3, 32, 32, 16]
        self.kern_sizes = kern_sizes  # kern_sizes = [5, 5, 5]
        self.hid_sizes = hid_sizes  # hid_sizes = [9216(16*24*24), 512]
        self.feat_size = feat_size  # feat_size = 64
        self.bottle_neck_image_size = None
        self.class_count = class_count  # class_count=27
        self.apply_classification_task = apply_classification_task
        self.random_seed = random_seed
        self.update_weights_method = update_weights_method

        self.apply_random_seed()

        # encoder
        self.L0_conv1 = nn.Conv2d(in_channels=chn_sizes[0], out_channels=chn_sizes[1], kernel_size=kern_sizes[0], stride=1, padding=0)
        self.L1_conv2 = nn.Conv2d(in_channels=chn_sizes[1], out_channels=chn_sizes[2], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L2_maxp1 = nn.MaxPool2d(kernel_size=2)
        self.L3_conv3 = nn.Conv2d(in_channels=chn_sizes[2], out_channels=chn_sizes[3], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L4_lenc1 = nn.Linear(in_features=hid_sizes[0], out_features=hid_sizes[1])
        self.L5_lenc2 = nn.Linear(in_features=hid_sizes[1], out_features=feat_size*2)

        # decoder
        self.L6_ldec2 = nn.Linear(in_features=feat_size, out_features=hid_sizes[1])
        self.L7_ldec1 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
        self.L8_dcnv3 = nn.ConvTranspose2d(in_channels=chn_sizes[3], out_channels=chn_sizes[2], kernel_size=kern_sizes[2], stride=1, padding=0)
        self.L9_upsm1 = nn.Upsample(scale_factor=2)
        self.L10_dcnv2 = nn.ConvTranspose2d(in_channels=chn_sizes[2], out_channels=chn_sizes[1], kernel_size=kern_sizes[1], stride=1, padding=0)
        self.L11_dcnv1 = nn.ConvTranspose2d(in_channels=chn_sizes[1], out_channels=chn_sizes[0], kernel_size=kern_sizes[0], stride=1, padding=0)

        # classifier
        if self.apply_classification_task:
            self.L8_lcls1 = nn.Linear(in_features=feat_size*2, out_features=hid_sizes[1])
            self.L9_lcls2 = nn.Linear(in_features=hid_sizes[1], out_features=hid_sizes[0])
            self.L10_sofm = nn.Linear(in_features=hid_sizes[0], out_features=self.class_count)

        print("input_size=", self.input_size)
        print("chn_sizes=", self.chn_sizes)
        print("kern_sizes=", self.kern_sizes)
        print("hid_sizes=", self.hid_sizes)
        print("feat_size=", self.feat_size)

        if self.update_weights_method is not None:
            self.update_weight_list = ["BCE", "KLD"]
        else:
            self.update_weight_list = None

        if self.apply_classification_task:
            print("class_count=", self.class_count)
            if self.update_weights_method is not None:
                self.update_weight_list.append("CLS")

        if self.update_weights_method is not None:
            self.update_weight_init = 1/len(self.update_weight_list)
            self.update_weight_min = 1/(len(self.update_weight_list)+1)

        lr = 0.0001
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_BCE = nn.BCELoss(reduction='sum')
        if self.apply_classification_task:
            self.loss_CLS = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_epoch_params = {
            "wBCE": 1/3,
            "wCLS": 1/3,
            "wKLD": 1/3,
            "sumBCE": None,
            "sumCLS": None,
            "sumKLD": None,
        }
        self.loss_prev_epoch = None

    def update_weights(self):
        if self.update_weights_method is None or self.loss_prev_epoch is None:
            return

        w_k_new_sum = 0
        for k in self.update_weight_list:
            sum_k_cur = self.loss_epoch_params["sum" + k]
            sum_k_prv = self.loss_prev_epoch["sum" + k]
            if sum_k_prv is None:
                print("skip update of weights - first epoch")
                return
            w_k_cur = self.loss_epoch_params["w" + k]
            print("w", k, "_cur=", w_k_cur)

            dif_k = sum_k_cur - sum_k_prv
            dif_k_p = dif_k/sum_k_cur

            w_k_new = np.maximum(w_k_cur * (1 + dif_k_p), self.update_weight_min)
            w_k_new_sum += w_k_new

            self.loss_epoch_params["w" + k] = w_k_new

        for k in self.update_weight_list:
            self.loss_epoch_params["w" + k] /= w_k_new_sum
            print("w", k, "_new=", self.loss_epoch_params["w" + k])

    def apply_random_seed(self):
        np.random.seed(self.random_seed )
        torch.manual_seed(self.random_seed )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def enc(self, x):
        # encoding
        x = F.relu(self.L0_conv1(x))
        x = F.relu(self.L1_conv2(x))
        x = self.L2_maxp1(x)
        x = F.relu(self.L3_conv3(x))
        if self.bottle_neck_image_size is None:
            self.bottle_neck_image_size = [x.size(1), x.size(2), x.size(3)]
        x = x.view(x.size(0), -1)
        x = F.relu(self.L4_lenc1(x))
        x = self.L5_lenc2(x)
        return x

    def dec(self, z):
        # decoding
        x = F.relu(self.L6_ldec2(z))
        x = F.relu(self.L7_ldec1(x))
        x = x.view(x.size(0), self.bottle_neck_image_size[0], self.bottle_neck_image_size[1], self.bottle_neck_image_size[2])
        x = F.relu(self.L8_dcnv3(x))
        x = self.L9_upsm1(x)
        x = F.relu(self.L10_dcnv2(x))
        reconstruction = torch.sigmoid(self.L11_dcnv1(x))
        return reconstruction

    def predict(self, x):
        x = F.relu(self.L8_lcls1(x))
        x = F.relu(self.L9_lcls2(x))
        x = self.L10_sofm(x)
        _, preds = torch.max(x, 1)
        return x, preds

    def forward(self, x):
        x = self.enc(x)

        if self.apply_classification_task:
            xProb, preds = self.predict(x)
        else:
            xProb, preds = None, None

        x = x.view(-1, 2, self.feat_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        reconstruction = self.dec(z)

        return reconstruction, mu, log_var, xProb, preds

    def final_loss(self, xProb, labels, reconstruction, data, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """

        if self.apply_classification_task:
            CLS = self.loss_CLS(xProb, labels)
        else:
            CLS=0
        BCE = self.loss_BCE(reconstruction, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self.loss_epoch_params["sumBCE"] += BCE.item()
        if self.apply_classification_task:
            self.loss_epoch_params["sumCLS"] += CLS.item()
        self.loss_epoch_params["sumKLD"] += KLD.item()
        return self.loss_epoch_params["wBCE"]*BCE + self.loss_epoch_params["wKLD"]*KLD + self.loss_epoch_params["wCLS"]*CLS

    def fit(self, X_data, batch_size):
        self.train()
        self.apply_random_seed()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        lab_vec = []
        pred_vec = []

        self.loss_prev_epoch = self.loss_epoch_params.copy()

        self.loss_epoch_params["sumBCE"] = 0
        self.loss_epoch_params["sumCLS"] = 0
        self.loss_epoch_params["sumKLD"] = 0

        for b in dloader:
            data = b['image']
            labels = b['label']

            data = data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            reconstruction, mu, logvar, xProb, preds = self.forward(data)

            lab_vec.append(labels)
            if self.apply_classification_task:
                pred_vec.append(preds)

            loss = self.final_loss(xProb, labels, reconstruction, data, mu, logvar)

            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        if self.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
            acc = accuracy_score(lab_vec, pred_vec)
        else:
            acc = 0

        self.update_weights()

        n = len(X_data)
        loss_acc_dict = {
            "loss": running_loss/n,
            "acc": acc,
            "wBCE": self.loss_epoch_params["wBCE"],
            "wCLS": self.loss_epoch_params["wCLS"],
            "wKLD": self.loss_epoch_params["wKLD"],
            "sumBCE": self.loss_epoch_params["sumBCE"]/n,
            "sumCLS": self.loss_epoch_params["sumCLS"]/n,
            "sumKLD": self.loss_epoch_params["sumKLD"]/n,
        }

        return loss_acc_dict

    def validate(self, X_vate, epoch, batch_size, out_folder, out_name_add_str=""):
        self.eval()
        self.apply_random_seed()

        running_loss = 0.0
        batch = [b['image'] for b in X_vate]
        batch_lb = [b['label'] for b in X_vate]
        uqlb, unid = np_unique(batch_lb, return_index=True)
        data_al = [batch[i] for i in unid]
        data_cn = len(unid)
        pred_vec = []

        self.loss_epoch_params["sumBCE"] = 0
        self.loss_epoch_params["sumCLS"] = 0
        self.loss_epoch_params["sumKLD"] = 0
        with torch.no_grad():
            fr = 0
            while (fr < len(X_vate)):
                to = fr + batch_size
                if to > len(X_vate):
                    to = len(X_vate)

                data = torch.stack(batch[fr:to], dim=0)
                labels = torch.tensor(batch_lb[fr:to])

                data = data.to(self.device)
                labels = labels.to(self.device)

                reconstruction, mu, logvar, xProb, preds = self.forward(data)

                if self.apply_classification_task:
                    pred_vec.append(preds)

                loss = self.final_loss(xProb, labels, reconstruction, data, mu, logvar)

                running_loss += loss.item()
                fr = to

        labels = np.asarray(batch_lb)
        if self.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
            acc = accuracy_score(labels, pred_vec)
        else:
            acc = 0

        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            reconstruction, _, _, _, _ = self.forward(data)
            both = torch.cat((data.view(data_cn, 3, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, 3, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_" + out_name_add_str + "{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)

        n = len(X_vate)
        loss_acc_dict = {
            "loss": running_loss/n,
            "acc": acc,
            "wBCE": self.loss_epoch_params["wBCE"],
            "wCLS": self.loss_epoch_params["wCLS"],
            "wKLD": self.loss_epoch_params["wKLD"],
            "sumBCE": self.loss_epoch_params["sumBCE"]/n,
            "sumCLS": self.loss_epoch_params["sumCLS"]/n,
            "sumKLD": self.loss_epoch_params["sumKLD"]/n,
        }
        return loss_acc_dict

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        model.apply_random_seed()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        mu_vec = []
        x_vec = []
        lab_vec = []
        pred_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b['image']
                labels = b['label']
                lab_vec.append(labels)

                x = data.to(device)
                # encode
                x = F.relu(model.L0_conv1(x))
                x = F.relu(model.L1_conv2(x))
                x = model.L2_maxp1(x)
                x = F.relu(model.L3_conv3(x))
                x = x.view(x.size(0), -1)
                x = F.relu(model.L4_lenc1(x))
                x = model.L5_lenc2(x)
                x_vec.append(x)

                if model.apply_classification_task:
                    _, preds = model.predict(x)
                    pred_vec.append(preds)

                x = x.view(-1, 2, model.feat_size)
                mu = x[:, 0, :]  # the first feature values as mean
                mu_vec.append(mu)

        mu_vec = np.asarray(torch.cat(mu_vec).to(torch.device('cpu')))
        x_vec = np.asarray(torch.cat(x_vec).to(torch.device('cpu')))
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        if model.apply_classification_task:
            pred_vec = np.asarray(torch.cat(pred_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', mu_vec=mu_vec, x_vec=x_vec, labsTr=lab_vec, predsTr=pred_vec)
        return mu_vec, x_vec, lab_vec, pred_vec

    def feat_extract(self, X_vate, batch_size):
        return self.feat_extract_ext(self, X_vate, batch_size)

class Sparsity_Loss_Base(nn.Module):
    def __init__(self):
        super(Sparsity_Loss_Base, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sparse_KL_DivergenceLoss(Sparsity_Loss_Base):
    def __init__(self, rho=None, reduction='batchmean'):
        super(Sparse_KL_DivergenceLoss, self).__init__()
        self.rho = rho
        self.rho_set = False
        self.reduction = reduction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_kl_rho_by_data(self, bt): #  bt: bottleneck
        if self.rho_set:
            return
        bt_dim = bt.shape[1]
        wanted_rho = self.rho
        self.rho = 1 / bt_dim
        if self.rho is None:
            print("self.kl_rho is set to {:f}, dimension of bottleneck is {:d}".format(1 / bt_dim, bt_dim))
        elif wanted_rho != 1/bt_dim:
            print("self.kl_rho changed from {:f} to {:f}, dimension of bottleneck is {:d}".format(wanted_rho, 1/bt_dim, bt_dim))
        self.rho_set = True

    def forward(self, bt):
        # https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/13
        # KLDLoss(p, q), sum(q) needs to equal one
        # p = log_softmax(tensor)
        self.set_kl_rho_by_data(bt)
        _bt = torch.sigmoid(bt)  # sigmoid because we need the probability distributions
        rho_mat = torch.tensor([self.rho] * np.ones(_bt.size()), dtype=torch.float32).to(self.device)
        loss_ret_1 = F.kl_div(F.log_softmax(_bt, dim=1).to(self.device), rho_mat, reduction=self.reduction)
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

class Conv_AE_NestedNamespace(nn.Module):
    def __init__(self, model_NestedNamespace):
        super(Conv_AE_NestedNamespace, self).__init__()

        # data related stuff
        self.input_size = model_NestedNamespace.INPUT_SIZE
        self.input_channel_size = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'INPUT_CHANNEL_SIZE', default_type=int, default_val=None)
        self.data_key = model_NestedNamespace.DATA_KEY

        # model related stuff
        self.model_name = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'MODEL_NAME', default_type=str, default_val='conv_ae_model')
        self.weight_decay = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'WEIGHT_DECAY', default_type=float, default_val=0.0)
        self.learning_rate = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'LEARNING_RATE', default_type=float, default_val=0.0001)
        self.encoder_list = get_encoder_from_ns(model_NestedNamespace.LAYERS.encoder)
        self.decoder_list = get_encoder_from_ns(model_NestedNamespace.LAYERS.decoder)
        # model_sub : reconstruction error
        self.recons_err_function = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'RECONSTRUCTION_ERROR_FUNCTION', default_type=str, default_val='MSE')
        self.recons_err_reduction = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'RECONSTRUCTION_ERROR_REDUCTION', default_type=str, default_val='mean')
        # model_sub : sparsity stuff
        self.sparsity_func = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'SPARSITY_ERROR', default_type=str, default_val=None)
        self.sparsity_weight = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'SPARSITY_WEIGHT', default_type=float, default_val=0.0)
        self.saprsity_reduction = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'SPARSITY_REDUCTION', default_type=str, default_val='mean')
        self.kl_rho = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'KL_DIV_RHO', default_type=float, default_val=0.05)
        # experiment reproducibility
        self.random_seed = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'RANDOM_SEED', default_type=int, default_val=7)
        # evaluation metrics related stuff
        self.bottleneck_act_apply = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'BOTTLENECK_ACT_APPLY', default_type=bool, default_val=None)
        self.bottleneck_kmeans_apply = funcH.get_attribute_from_nested_namespace(model_NestedNamespace, 'BOTTLENECK_KMEANS_APPLY', default_type=bool, default_val=False)

        self.__init_stuff__()
        self.apply_random_seed()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_model_def()

        self.export_image_ids_dict = {}
        self.bottleneck_fig = {
            'save_fold_name': None,
            'save_fig_name_base': model_NestedNamespace.BOTTLENECK_FIG_NAME_BASE if 'BOTTLENECK_FIG_NAME_BASE' in vars(model_NestedNamespace) else None,
            'save_fig_name': model_NestedNamespace.BOTTLENECK_FIG_NAME_BASE if 'BOTTLENECK_FIG_NAME_BASE' in vars(model_NestedNamespace) else None,
        }

    def __init_stuff__(self):
        self.__init_model_setting__()
        self.__init_error_stuff__()
        self.__init_clustering_stuff__()
    def __init_model_setting__(self):
        for layer_name in self.encoder_list:
            print(layer_name, self.encoder_list[layer_name])
            setattr(self, layer_name, self.encoder_list[layer_name]['layer_module'])

        for layer_name in self.decoder_list:
            print(layer_name, self.decoder_list[layer_name])
            if layer_name == "flat_b" and self.decoder_list[layer_name]['layer_type'] == 'Unflatten':
                setattr(self, layer_name, getattr(self, 'flat'))
            else:
                setattr(self, layer_name, self.decoder_list[layer_name]['layer_module'])
    def __init_error_stuff__(self):
        reconstruction_err_func = nn.MSELoss(reduction=self.recons_err_reduction)
        if self.recons_err_function == 'BCE':
            reconstruction_err_func = nn.BCELoss(reduction=self.recons_err_reduction)

        if self.sparsity_func is None:
            bottleneck_func = None
        elif self.sparsity_func in ['kl_divergence', 'kl_divergence_sigmoid']:
            bottleneck_func = Sparse_KL_DivergenceLoss(rho=self.kl_rho, reduction=self.saprsity_reduction)
        elif self.sparsity_func == 'l1_norm':
            bottleneck_func = Sparse_Loss_Dim(dim=1, reduction=self.saprsity_reduction)
        elif self.sparsity_func == 'l2_norm':
            bottleneck_func = Sparse_Loss_Dim(dim=2, reduction=self.saprsity_reduction)
        elif self.sparsity_func == 'cross_entropy':
            bottleneck_func = Sparse_Loss_CrossEntropy(reduction=self.saprsity_reduction, apply_sigmoid_activation=False)
        elif self.sparsity_func == 'cross_entropy_with_sigmoid':
            bottleneck_func = Sparse_Loss_CrossEntropy(reduction=self.saprsity_reduction, apply_sigmoid_activation=True)

        self.loss = {
            'reconstruction': {'func': reconstruction_err_func, 'val': None},
            'sparsity': {'func': bottleneck_func, 'val': None},
        }
    def __init_clustering_stuff__(self):
        self.clustering_decided = None
        self.cluster_any = None
        self.clustering_dict = {
            'bottleneck_kmeans': {'apply': self.bottleneck_kmeans_apply, 'val': None},
            'bottleneck_act': {'apply': self.bottleneck_act_apply, 'val': None},
        }

    def print_model_def(self):
        print("data_key : ", self.data_key)
        print("model_name : ", self.model_name)
        print("learning_rate : ", self.learning_rate)
        print("random_seed : ", self.random_seed)
        print("optimizer : ", self.optimizer)
        print("encoder_list : ")
        funcH.print_params_nested_namespace(self.encoder_list)
        print("decoder_list : ")
        funcH.print_params_nested_namespace(self.decoder_list)
        print("loss : ", self.loss)
        print("reconstruction.func : ", self.recons_err_function)
        print("reconstruction.func.reduction : ", self.recons_err_reduction)
        print("sparsity.func : ", self.sparsity_func)
        print("sparsity_weight : ", self.sparsity_weight)
        print("kl_rho : ", self.kl_rho)
        print("saprsity_reduction : ", self.saprsity_reduction)
        print("bottleneck_act.apply : ", self.bottleneck_act_apply)
        print("bottleneck_kmeans.apply : ", self.bottleneck_kmeans_apply)

    def apply_random_seed(self):
        np.random.seed(self.random_seed )
        torch.manual_seed(self.random_seed )

    def enc(self, x):
        for layer_name in self.encoder_list:
            f = getattr(self, layer_name)
            x = f(x)
        return x

    def dec(self, x):
        for layer_name in self.decoder_list:
            f = getattr(self, layer_name)
            if layer_name=='flat_b':
                x = f.unflatten(x)
            else:
                x = f(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction

    def forward(self, x):
        bottleneck = self.enc(x)
        reconstruction = self.dec(bottleneck)
        return reconstruction, bottleneck

    def final_loss(self, data, reconstruction, bottleneck):
        loss_reconstruction = self.loss['reconstruction']['func'](reconstruction, data)
        self.loss['reconstruction']['val'] += loss_reconstruction.item()

        loss_sparsity = 0.0
        if self.sparsity_weight is not None and self.sparsity_weight > 0.0:
            loss_sparsity = -self.sparsity_weight*self.loss['sparsity']['func'](bottleneck)
            self.loss['sparsity']['val'] += loss_sparsity.item()/self.sparsity_weight

        return loss_reconstruction + loss_sparsity #sum(self.loss[i]['val'] for i in self.loss)

    def reset_loss_vals(self):
        for i in self.loss:
            self.loss[i]['val'] = 0.0

    def clustering_decide(self, X_data):
        #here we will check a couple of things
        initial_sample = X_data[0]
        label_exist = 'label' in initial_sample
        sparsity_applied = self.sparsity_weight is not None and self.sparsity_weight > 0.0
        self.clustering_dict['bottleneck_act']['apply'] = self.clustering_dict['bottleneck_act']['apply'] or (label_exist and sparsity_applied)
        self.clustering_decided = True

        self.cluster_any = False
        for k in self.clustering_dict:
            self.cluster_any = self.cluster_any or self.clustering_dict[k]['apply']

        return label_exist

    def cluster_bottleneck(self, lab_vec, bottleneck_vec):
        if not self.cluster_any:
            return
        for k in self.clustering_dict:
            if not self.clustering_dict[k]['apply']:
                continue
            if k == 'bottleneck_kmeans':
                print('bottleneck_kmeans')
                pred_vec, kc_tr = funcH.clusterData(bottleneck_vec, n_clusters=bottleneck_vec.shape[1], normMode='', applyPca=False, clusterModel='KMeans', verbose=0)
                centroid_info_pdf = funcH.get_cluster_centroids(bottleneck_vec, pred_vec, kluster_centers=kc_tr, verbose=0)
            if k == 'bottleneck_act':
                print('bottleneck_act')
                pred_vec = np.argmax(bottleneck_vec.T, axis=0).T.squeeze()
                centroid_info_pdf = funcH.get_cluster_centroids(bottleneck_vec, pred_vec, kluster_centers=None, verbose=0)
            _confMat_preds, _, kr_pdf, weightedPurity, cnmxh_perc = funcH.countPredictionsForConfusionMat(lab_vec, pred_vec, centroid_info_pdf=centroid_info_pdf, labelNames=None)
            sampleCount = np.sum(np.sum(_confMat_preds))
            acc = 100 * np.sum(np.diag(_confMat_preds)) / sampleCount
            print("--acc({:5.3f}), uniqClustCnt({:d})".format(acc, np.size(np.unique(pred_vec))))
            self.clustering_dict[k]['val'] = acc

    def apply_acc(self, loss_dict, lab_vec, bottleneck_vec):
        if not self.cluster_any:
            return loss_dict
        lab_vec = np.asarray(torch.cat(lab_vec).to(torch.device('cpu')))
        bottleneck_vec = np.asarray(torch.cat(bottleneck_vec).to(torch.device('cpu')).detach().numpy())
        num_samples, cluster_count = bottleneck_vec.shape
        rand_sample_ids = np.sort(np.random.permutation(np.arange(num_samples))[:int(cluster_count*1.5)])

        if self.bottleneck_fig['save_fold_name'] is not None and self.bottleneck_fig['save_fig_name'] is not None:
            ax = plt.imshow(bottleneck_vec[rand_sample_ids, :], cmap='hot', interpolation='nearest')
            fig = ax.get_figure()
            plt.colorbar()
            fig.savefig(os.path.join(self.bottleneck_fig['save_fold_name'], self.bottleneck_fig['save_fig_name']), bbox_inches='tight', dpi=600)

            var_dim = np.var(bottleneck_vec, axis=0)
            plt.clf()
            fig, ax = plt.subplots(1, figsize=(15, 8), dpi=80)
            title_str = 'min_var(' + str(np.min(var_dim)) + '),max_var(' + str(np.max(var_dim)) + ')'
            ax.plot(np.asarray(range(0, len(var_dim))), var_dim.squeeze(), lw=2, label='variance', color='red')
            ax.set_title(title_str)
            fig.savefig(os.path.join(self.bottleneck_fig['save_fold_name'], self.bottleneck_fig['save_fig_name'].replace('.png','_var.png')), bbox_inches='tight', dpi=80)

            # the histogram of the data
            plt.clf()
            fig, ax = plt.subplots(1, figsize=(15, 8), dpi=80)
            pred_vec = np.argmax(bottleneck_vec.T, axis=0).squeeze()
            to_plot = bottleneck_vec[np.asarray(range(0, bottleneck_vec.shape[0]), dtype=int), pred_vec]
            # n, bins, patches = plt.hist(to_plot, np.asarray(np.linspace(0.1, 1, 10)), density=False)
            n, bins, patches = plt.hist(to_plot, bins=10, density=False)
            plt.xlabel('Bins')
            plt.ylabel('Softmax(Activation)')
            plt.title('Winner Bottleneck Activation Histogram')
            for i in range(0, len(patches)):
                if i < len(patches)-1:
                    x_pos = (patches[i].xy[0] + patches[i+1].xy[0])/2
                else:
                    x_pos = patches[i].xy[0]
                height = n[i]
                plt.text(x_pos, height/2, r'$'+str(height)+'$')
            # plt.text(0.5, np.max(n)*0.75, r'less than 0.1 :$' + str(int(len(pred_vec)-np.sum(n))) + '$')
            plt.xlim(np.min(bins), np.max(bins))  # plt.xlim(0.1, 1.0)
            plt.xticks(bins)
            plt.grid(True)
            fig.savefig(os.path.join(self.bottleneck_fig['save_fold_name'],
                                     self.bottleneck_fig['save_fig_name'].replace('.png', '_hist.png')),
                                     bbox_inches='tight')

            plt.close('all')

        self.cluster_bottleneck(lab_vec, bottleneck_vec)
        for k in self.clustering_dict:
            if not self.clustering_dict[k]['apply']:
                continue
            loss_dict[k] = self.clustering_dict[k]['val']

        return loss_dict

    def fit(self, X_data, batch_size, out_folder=None, epoch=None):
        self.train()
        self.apply_random_seed()
        running_loss = 0.0
        dloader = DataLoader(X_data, batch_size=batch_size, shuffle=True)
        self.reset_loss_vals()

        self.clustering_decide(X_data)
        if self.cluster_any:
            lab_vec = []
            bottleneck_vec = []

        for b in dloader:
            data = b[self.data_key]
            data = data.to(self.device)
            if self.cluster_any:
                labels = b['label']
                lab_vec.append(labels)

            self.optimizer.zero_grad()
            reconstruction, bottleneck = self.forward(data)
            loss = self.final_loss(data, reconstruction, bottleneck)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if self.cluster_any:
                bottleneck_vec.append(bottleneck)

        n = len(X_data)
        loss_dict = {"running loss": running_loss/n}
        for i in self.loss:
            loss_dict[i] = self.loss[i]['val']

        self.setup_bottleneck_heatmap(out_folder, epoch, sub_data_identifier='fit')

        loss_dict = self.apply_acc(loss_dict, lab_vec, bottleneck_vec)

        return loss_dict

    def fill_export_image_ids_dict(self, X_vate, sub_data_identifier):
        initial_sample = X_vate[0]
        if 'label' in initial_sample:
            print('__finding images to print from labels')
            batch_lb = [b['label'] for b in X_vate]
            uqlb, unid = np_unique(batch_lb, return_index=True)
        else:
            print('__finding images to print randomly')
            unid = np.random.permutation(np.arange(len(X_vate)))[:10]
        self.export_image_ids_dict[sub_data_identifier] = unid

    def setup_bottleneck_heatmap(self, out_folder, epoch, sub_data_identifier=""):
        if out_folder is not None and epoch is not None:
            self.bottleneck_fig['save_fold_name'] = out_folder
            self.bottleneck_fig['save_fig_name'] = str(self.bottleneck_fig['save_fig_name_base']).replace("XXX", str(epoch).zfill(3))
            self.bottleneck_fig['save_fig_name'] = str(self.bottleneck_fig['save_fig_name']).replace(".png", "_" + sub_data_identifier + ".png")

    def validate(self, X_vate, epoch, batch_size, out_folder, sub_data_identifier):
        self.eval()
        self.apply_random_seed()

        lab_vec = []
        bottleneck_vec = []

        running_loss = 0.0
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)
        self.reset_loss_vals()
        with torch.no_grad():
            for b in dloader:
                data = b[self.data_key]
                data = data.to(self.device)
                if self.cluster_any:
                    labels = b['label']
                    lab_vec.append(labels)
                reconstruction, bottleneck = self.forward(data)
                loss = self.final_loss(data, reconstruction, bottleneck)
                running_loss += loss.item()
                if self.cluster_any:
                    bottleneck_vec.append(bottleneck)

        if sub_data_identifier not in self.export_image_ids_dict:
            self.fill_export_image_ids_dict(X_vate, sub_data_identifier)
        data_al = [X_vate[i][self.data_key] for i in self.export_image_ids_dict[sub_data_identifier]]
        data_cn = len(self.export_image_ids_dict[sub_data_identifier])
        with torch.no_grad():
            # save the last batch input and output of every epoch
            data = torch.stack(data_al, dim=0)
            data = data.to(self.device)
            reconstruction, _ = self.forward(data)
            both = torch.cat((data.view(data_cn, self.input_channel_size, self.input_size, self.input_size)[:data_cn],
                              reconstruction.view(data_cn, self.input_channel_size, self.input_size, self.input_size)[:data_cn]))
            f_name = out_folder + "/output_" + sub_data_identifier + "{:03d}.png".format(epoch)
            save_image(both.cpu(), f_name, nrow=data_cn)

        n = len(X_vate)
        loss_dict = {"valid loss": running_loss/n}
        for i in self.loss:
            loss_dict[i] = self.loss[i]['val']

        self.setup_bottleneck_heatmap(out_folder, epoch, sub_data_identifier=sub_data_identifier)

        loss_dict = self.apply_acc(loss_dict, lab_vec, bottleneck_vec)

        return loss_dict

    @staticmethod
    def feat_extract_ext(model, X_vate, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, str):
            model = torch.load(model, map_location=device)
        model.eval()
        model.apply_random_seed()
        dloader = DataLoader(X_vate, batch_size=batch_size, shuffle=False)

        feat_vec = []
        with torch.no_grad():
            for b in dloader:
                data = b[model.data_key]
                data = data.to(device)
                x = data.to(device)
                # encode
                x = model.enc(x)
                feat_vec.append(x.to(torch.device('cpu')))

        feat_vec = np.asarray(torch.cat(feat_vec).to(torch.device('cpu')))
        #np.savez('?_data.npz', feat_vec=feat_vec)
        return feat_vec

    def feat_extract(self, X_vate, batch_size):
        return self.feat_extract_ext(self, X_vate, batch_size)
