import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()

        # extract model settings from args
        self.args = args
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type
        self.gen_hiddens = args.gen_hiddens
        self.args.dropout=0.2
        self.args.gen_depth=2#
        self.CUDA_DEVICE=args.CUDA_DEVICE

        if self.input_size ==torch.Size( [3, 56, 56] ):
            self.last_kernel_size = 7
        if self.input_size == torch.Size([3, 224, 224]):
            self.last_kernel_size = 28
        elif self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
            self.last_kernel_size = 7
        elif self.input_size == [1, 28, 20]:
            self.last_kernel_size = (7, 5)
        elif self.input_size == [3, 32, 32]:
            self.last_kernel_size = 8
        elif len(self.input_size)==1:#hidden features
            self.last_kernel_size = 8
        else:

            raise ValueError('invalid input size!!')

        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()

        self.q_z_nn_output_dim = 256

        # auxiliary

        self.FloatTensor = torch.FloatTensor

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

        self.prior = MultivariateNormal(torch.zeros(args.z_size), torch.eye(args.z_size))

        # get gradient dimension:
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """

        if self.input_type == 'binary':

            if self.args.gen_architecture == 'GatedConv':
                q_z_nn = nn.Sequential(
                    GatedConv2d(self.input_size[0], 32, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(32, 32, 5, 2, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(32, 64, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, 64, 5, 2, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, 64, 5, 1, 2),
                    nn.Dropout(self.args.dropout),
                    GatedConv2d(64, self.gen_hiddens, self.last_kernel_size, 1, 0),
                )
                assert self.args.gen_depth == 6

            elif self.args.gen_architecture == 'MLP':#I can use this and then use the hidden features of the model as inptut!
                q_z_nn = [
                    Reshape([-1]),
                    nn.Linear(np.prod(self.args.input_size), self.gen_hiddens),
                    nn.ReLU(True),
                    nn.Dropout(self.args.dropout),
                ]
                for i in range(1, self.args.gen_depth):
                    q_z_nn += [
                        nn.Linear(self.args.gen_hiddens, self.args.gen_hiddens),
                        nn.ReLU(True),
                        nn.Dropout(self.args.dropout),
                    ]
                q_z_nn = nn.Sequential(*q_z_nn)

            q_z_mean = nn.Linear(self.gen_hiddens, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(self.gen_hiddens, self.z_size),
                nn.Softplus(),
            )
            return q_z_nn, q_z_mean, q_z_var

        #TODO(add log_logistic loss for continuous)
        elif self.input_type in ['multinomial', 'continuous']:
            act = None

            q_z_nn = nn.Sequential(
                GatedConv2d(self.input_size[0], 32, 4, 2, 1, activation=act),
                GatedConv2d(32, 32, 5, 2, 2, activation=act),
                GatedConv2d(32, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 64, 5, 2, 2, activation=act),
                GatedConv2d(64, 64, 5, 1, 2, activation=act),
                GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            )
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(
                nn.Linear(256, self.z_size),
                nn.Softplus(),
                nn.Hardtanh(min_val=0.01, max_val=7.)

            )
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """




        if self.input_type == 'binary':
            if self.args.gen_architecture == 'GatedConv':
                p_x_nn = nn.Sequential(
                    Reshape([self.args.z_size, 1, 1]),
                    GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0),
                    GatedConvTranspose2d(64, 64, 5, 1, 2),
                    GatedConvTranspose2d(64, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2),
                    GatedConvTranspose2d(32, 32, 5, 2, 2, 1),
                    GatedConvTranspose2d(32, 32, 5, 1, 2)
                )
                p_x_mean = nn.Sequential(
                    nn.Conv2d(32, self.input_size[0], 1, 1, 0),
                    nn.Sigmoid()
                )
            elif self.args.gen_architecture=='MLP':
                p_x_nn = [
                    nn.Linear(self.z_size, self.gen_hiddens),
                    nn.ReLU(True),
                    nn.Dropout(self.args.dropout),
                ]
                for i in range(1, self.args.gen_depth):
                    p_x_nn += [
                        nn.Linear(self.args.gen_hiddens, self.args.gen_hiddens),
                        nn.ReLU(True),
                        nn.Dropout(self.args.dropout),
                    ]
                p_x_nn = nn.Sequential(*p_x_nn)

                p_x_mean = nn.Sequential(
                    nn.Linear(self.args.gen_hiddens, np.prod(self.args.input_size)),
                    nn.ReLU(True),#that was Sigmoid
                    Reshape(self.args.input_size)
                )
            return p_x_nn, p_x_mean

        #TODO(add log_logistic loss for continuous)
        elif self.input_type in ['multinomial', 'continuous']:
            act = None
            p_x_nn = nn.Sequential(
                Reshape([self.args.z_size, 1, 1]),
                GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act),
                GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
                GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
                GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
                GatedConvTranspose2d(32, 32, 4, 2, 1, activation=act)
            )
            # GatedConv2d(self.input_size[0], 32, 4, 2, 1, activation=act),
            # GatedConv2d(32, 32, 5, 2, 2, activation=act),
            # GatedConv2d(32, 64, 5, 1, 2, activation=act),
            # GatedConv2d(64, 64, 5, 2, 2, activation=act),
            # GatedConv2d(64, 64, 5, 1, 2, activation=act),
            # GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act)
            p_x_mean = nn.Sequential(
                nn.Conv2d(32, 256, 5, 1, 2),
                nn.Conv2d(256, self.input_size[0] , 1, 1, 0),

            )

            return p_x_nn, p_x_mean

        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps =eps.to(self.CUDA_DEVICE)
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """

        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)

        return mean, var


    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def generate(self, N=16):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """

        z = self.prior.sample((N,))
        if self.args.cuda: z = z.to(self.args.device)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)

        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """

        # mean and variance of z

        z_mu, z_var = self.encode(x)
        # sample z
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)

        return x_mean, z_mu, z_var, self.log_det_j, z, z

def calculate_loss(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=1.,reduction="sum"):
    """
    Picks the correct loss depending on the input type.
    """

    loss, rec, kl = mse_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta,reduction=reduction)
    bpd = 0.


    return loss, rec, kl, bpd

def mse_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.,reduction="sum"):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    batch_size = x.size(0)

    # - N E_q0 [ ln p(x|z_k) ]
    #bce = reconstruction_function(recon_x, x)
    re = F.mse_loss(recon_x, x,reduction=reduction)
    log_var=z_var.log()

    kl= -0.5 * torch.sum(1 + log_var - z_mu.pow(2) - log_var.exp())
    if 0:
        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_standard(z_k, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = torch.sum(log_q_z0 - log_p_zk)

        # sum over batches
        summed_ldj = torch.sum(ldj)

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        kl = (summed_logs - summed_ldj)
        if reduction=="sum" and len(list(x.size()))>2:
            re=re/np.prod((list(x.size())[1:3])) #to be generalized


    #loss = loss / float(batch_size)
    if reduction!="mean":
        re = re / float(batch_size)
    kl = kl / float(batch_size)
    loss = re + beta * kl
    return loss, re, kl


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm

def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):

    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


class GatedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g

class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):

        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)