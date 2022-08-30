import math
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import torchvision.models as models


class CND_Module(nn.Module):

    def __init__(self, input_sizes, hidden_size, num_ftrs, final_output, depth=2):
        super(CND_Module, self).__init__()
        temp_layers = []
        for i in range(len(input_sizes)):
            temp_layers.append(nn.Sequential(nn.Linear(input_sizes[i], 2 * hidden_size),
                                             nn.ReLU(True)))
        self.input_layers = nn.ModuleList(temp_layers)

        self.second_layer = nn.Sequential(nn.Linear(2 * hidden_size, int(hidden_size)),
                                          nn.ReLU(True))
        self.feat_layer = nn.Sequential(nn.Linear(int(hidden_size), num_ftrs),
                                        )
        self.proj_layer = nn.Sequential(nn.Linear(num_ftrs, int(num_ftrs / 2)),
                                        nn.ReLU(True))
        self.final_layer = nn.Sequential(nn.Linear(int(num_ftrs / 2), final_output),
                                         )

    def train_forward(self, xs):
        out = 0

        for i in range(len(xs)):
            xv = xs[i].view(xs[i].size(0), -1)
            out += self.input_layers[i](xv)

        out = self.second_layer(out)
        out = self.feat_layer(out)
        out = F.normalize(out, p=2, dim=1)
        proj = self.proj_layer(out)
        proj = self.final_layer(proj)
        # roj = F.normalize(proj, p=2, dim=1)
        # noproj_out = F.normalize(out, p=2, dim=1)
        # out = F.relu(out)
        return out, proj# out is the same space were softmax is applied, projection is the learned space of the ND module

    def forward(self, xs):
        out = 0

        for i in range(len(xs)):
            xv = xs[i].view(xs[i].size(0), -1)
            out += self.input_layers[i](xv)

        out = self.second_layer(out)
        out = self.feat_layer(out)
        out = F.normalize(out, p=2, dim=1)
        # out = self.proj_layer(out)
        # out = self.final_layer(out)
        # out = F.normalize(out, p=2, dim=1)
        # out = F.relu(out)
        return out

    def target_forward(self, x):

        x = F.normalize(x, p=2, dim=1)
        # out = self.proj_layer(x)
        # out = self.final_layer(out)
        # out = F.normalize(out, p=2, dim=1)
        # out = F.relu(out)
        out = x
        return out


class CVAE(nn.Module):
    def __init__(self, d, args, **kwargs):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            # nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True),

            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.f = 4  # 8
        self.d = d

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return h1
        # return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar
        hid = self.encode(x)
        return self.decode(hid), hid

    def sample(self, size):
        sample = Variable(torch.randn(size, self.d * self.f ** 2), requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


# Classifiers
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CategoricalConditionalBatchNorm(torch.nn.Module):
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            # weight = self.weight.index_select(0, cats).view(shape)
            weight = self.weight[cats].view(1, -1, 1, 1)  # .expand_as(shape)
            bias = self.bias[cats].view(1, -1, 1, 1)
            # bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()

        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        # self.bn1  = CategoricalConditionalBatchNorm(nf, 2)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion  # if input_size[1] in [8,16,21,32,42] else 640
        self.classifier = nn.Sequential(nn.Linear(last_hid, num_classes))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        # pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        # post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        # out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.classifier(out)
        return out


class ResNet_dropout(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size, dropr):
        super(ResNet_dropout, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        # self.bn1  = CategoricalConditionalBatchNorm(nf, 2)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(dropr)
        # hardcoded for now
        last_hid = nf * 8 * block.expansion  # if input_size[1] in [8,16,21,32,42] else 640

        self.classifier = nn.Sequential(
            nn.Linear(last_hid, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        # pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        # post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        # out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.classifier(out)
        return out


def ResNet18(nclasses, nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size)


def ResNet18_dropout(nclasses, nf=20, input_size=(3, 32, 32), dropr=0.5):
    return ResNet_dropout(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size, dropr=dropr)


# ========================Imprint=======================================
class ResNet18_imprint(nn.Module):

    def __init__(self, n_classes=200, normalize=True, scale=True, d_emb=200, nf=20, input_size=(3, 32, 32)):
        super().__init__()
        self.n_classes = n_classes
        self.d_emb = d_emb
        self.normalize = normalize
        self.scale = scale
        self.last_hid = nf * 8 * BasicBlock.expansion
        self.extractor = Extractor(nf, input_size)
        self.embedding = Embedding(self.last_hid, d_emb, normalize)
        if n_classes > 0:
            # to coope with multi head
            self.classifier = nn.Sequential(Classifier(n_classes, d_emb, normalize))
        if scale:
            # self.s = nn.Parameter(torch.FloatTensor([10]))
            # to coope with multi head
            self.classifier._modules["0"].s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):

        x = self.extractor(x)
        x = self.embedding(x)
        x = self.classifier(x)
        # to coope with multi head
        # if self.scale: x = self.s * x#duplicated
        return x

    def return_hidden(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'n_classes={n_classes}, ' \
                    'd_emb={d_emb}, ' \
                    'normalize={normalize}, ' \
                    'scale={scale}, '.format(**self.__dict__)
        return extra_str


class Extractor(nn.Module):

    def __init__(self, nf=20, input_size=(3, 32, 32)):
        super().__init__()
        num_blocks = [2, 2, 2, 2]
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(BasicBlock, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        self.last_hid = nf * 8 * BasicBlock.expansion  # if input_size[1] in [8,16,21,32,42] else 640

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class Embedding(nn.Module):

    def __init__(self, last_hid, d_emb=64, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.d_emb = d_emb
        self.fc = nn.Linear(last_hid, d_emb)

    def forward(self, x):
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'd_emb={d_emb}, ' \
                    'normalize={normalize}, '.format(**self.__dict__)
        return extra_str


class Classifier(nn.Module):

    def __init__(self, n_classes, d_emb=64, normalize=True):
        super().__init__()
        self.n_classes = n_classes
        self.in_features = d_emb
        self.normalize = normalize
        self.fc = nn.Linear(d_emb, n_classes, bias=False)

    def forward(self, x):
        if self.normalize:
            w = self.fc.weight
            w = F.normalize(w, dim=1, p=2)
            x = F.linear(x, w)
            # to coope with multihead
            if hasattr(self, "s"):
                x = self.s * x
        else:
            x = self.fc(x)
        return x

    def extra_repr(self):
        # include model specific parameters
        extra_str = 'n_classes={n_classes}, ' \
                    'd_emb={d_emb}, ' \
                    'normalize={normalize}, '.format(**self.__dict__)
        return extra_str


# ========================Emprint===================================
class MLP(nn.Module):
    def __init__(self, args, num_classes=10, nf=400):
        super(MLP, self).__init__()

        self.input_size = np.prod(args.input_size)
        self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
                                    nn.ReLU(True),
                                    nn.Linear(nf, nf),
                                    nn.ReLU(True))

        self.linear = nn.Linear(nf, num_classes)

    def return_hidden(self, x):
        x = x.view(-1, self.input_size)
        return self.hidden(x)

    def forward(self, x):
        out = self.return_hidden(x)
        return self.linear(out)


class MLP_dropout(nn.Module):
    def __init__(self, args, num_classes=10, nf=400):
        super(MLP_dropout, self).__init__()

        self.input_size = np.prod(args.input_size)
        self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
                                    nn.ReLU(True),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(nf, nf),
                                    nn.ReLU(True),
                                    nn.Dropout(p=0.2))

        self.linear = nn.Linear(nf, num_classes)

    def return_hidden(self, x):
        x = x.view(-1, self.input_size)
        return self.hidden(x)

    def forward(self, x):
        out = self.return_hidden(x)
        return self.linear(out)


class ResNet_PreTrained(nn.Module):

    def __init__(self, n_classes):
        super(ResNet_PreTrained, self).__init__()

        self.basenet = models.resnet18(pretrained=True)
        self.extractor = nn.Sequential(*list(self.basenet.children())[:-1])
        self.basenet.fc = torch.nn.Linear(self.basenet.fc.in_features, n_classes)
        self.linear = self.basenet.fc

    def forward(self, x):
        x = self.basenet(x)

        return x

    def return_hidden(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    '16Slim': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    '11Slim': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512]

}


class VGGSlim(models.VGG):

    def __init__(self, config='11Slim', num_classes=50, init_weights=True):
        features = make_layers(cfg[config])
        super(VGGSlim, self).__init__(features)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, num_classes),
        )
        if init_weights:
            self._initialize_weights()


class VGGSlim2(models.VGG):

    def __init__(self, config='11Slim', num_classes=50, init_weights=True):
        features = make_layers(cfg[config])
        super(VGGSlim2, self).__init__(features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True))
        self.classifier = nn.Sequential(

            nn.Linear(500, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def return_hidden(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.return_hidden(x)
        x = self.classifier(x)
        return x


import torch.nn as nn
import torchvision.models as models


class alex_seq(nn.Module):

    def __init__(self, pretrained, num_classes, dropr=0):
        super(alex_seq, self).__init__()

        self.basenet = models.alexnet(pretrained=pretrained)
        self.features = self.basenet.features  # nn.Sequential(*list(self.basenet.children())[:-1])
        self.avgpool = self.basenet.avgpool
        self.fc = nn.Sequential(*list(self.basenet.classifier.children())[:-1])
        for name, module in self.fc.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropr
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
        del self.basenet

    def return_hidden(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.return_hidden(x)
        x = self.classifier(x)
        return x


class VGGSlim2_dropout(models.VGG):

    def __init__(self, config='11Slim', num_classes=50, init_weights=True, dropr=0.5):
        features = make_layers(cfg[config])
        super(VGGSlim2_dropout, self).__init__(features)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 500),
            nn.ReLU(True), nn.Dropout(dropr),
            nn.Linear(500, 500),
            nn.ReLU(True), nn.Dropout(dropr))
        self.classifier = nn.Sequential(

            nn.Linear(500, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def return_hidden(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.return_hidden(x)
        x = self.classifier(x)
        return x