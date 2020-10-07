#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from numpy import unique as np_unique
from torch.utils.data import DataLoader
import numpy as np

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
