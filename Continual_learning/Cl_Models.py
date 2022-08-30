from __future__ import print_function, division

import torch
import torch.nn as nn
import copy
from Continual_learning.Architecture import  ResNet18_imprint,Classifier
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision
from torchvision import  models, transforms
import sys

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
# sys.path.append('../my_utils')
# from ImageFolderTrainVal import *
import pdb


class Multihead_Model(nn.Module):
    """
    Model for incremental learning with multiple heads each for a task
    """
    def __init__(self, module):
        super(Multihead_Model, self).__init__()
        self.module = module


        self.num_ftrs=self.module.num_ftrs
        del(self.module.num_ftrs)
        self.last_layer_index = self.module.last_layer_index
        del (self.module.last_layer_index)

    def forward(self, x):

        x=self.module.return_hidden(x)
        outputs=[]
        for namex, modulex in self.module.classifier._modules.items():


            outputs.append(modulex(x))




        return outputs


class Sharedhead_Model(nn.Module):
    """
     Model for incremental learning with shared head among all tasks
    """
    def __init__(self, module):
        super(Sharedhead_Model, self).__init__()
        self.module = module


        self.num_ftrs=self.module.num_ftrs
        del(self.module.num_ftrs)
        self.last_layer_index = self.module.last_layer_index
        del (self.module.last_layer_index)

    def forward(self, x):

        x=self.module.return_hidden(x)
        outputs=[]
        #this must be a list of one iterm. It is a wrapper to work for both model types, shared and multi-head
        for namex, modulex in self.module.classifier._modules.items():


            outputs.append(modulex(x))


        return outputs


def prepare_cl_model( model,dset_classes,task_index,opt):

    if opt.shared_head:
        if not (type(model) is Sharedhead_Model):
            model = Sharedhead_Model(model)
        original_model = copy.deepcopy(model)
        if not opt.all_initialized:
            model = expand_sharedhead_model(model, task_index, dset_classes,opt)# TO test initializing all neurons first!
    else:
        if not (type(model) is Multihead_Model):
            model = Multihead_Model(model)
        original_model = copy.deepcopy(model)
        model=expand_multihead_model(model, task_index, dset_classes,opt)

    return model,original_model

def expand_multihead_model(model,task_index,dset_classes,opt):
    """
    add a new head to multi-head model
    :param model:
    :param task_index:
    :param dset_classes:
    :return:
    """
    if task_index > 1:#not the first task
        if isinstance( model.module,ResNet18_imprint):#check of other network
            model.module.classifier.add_module(str(len(model.module.classifier._modules)), Classifier( dset_classes,model.num_ftrs))
        else:

            model.module.classifier.add_module(str(len(model.module.classifier._modules)),
                                               nn.Linear(model.num_ftrs, dset_classes,bias=not(opt.no_bias)))
    #watch out! this was not commented and the layer was initialize twice!.
    else:
        model.module.classifier._modules[model.last_layer_index] = nn.Linear(model.num_ftrs, dset_classes, bias=not(opt.no_bias))
        #     model.module.classifier._modules[model.last_layer_index] = nn.Linear(model.num_ftrs, dset_classes)

    return model


def expand_sharedhead_model(model,task_index,dset_classes,opt):
    """
    expand the number of neurons in a shared head output layer
    :param model:
    :param task_index:
    :param dset_classes:
    :return:
    """
    cuda_device = next(model.parameters()).device
    bias=not(opt.no_bias)#not isinstance(model.module,ResNet18_imprint)
    if task_index > 1:#not the first task
        if isinstance(model.module, ResNet18_imprint):
            old_classes=model.module.classifier._modules[model.last_layer_index].fc.out_features
            pre_out_layer = model.module.classifier._modules[model.last_layer_index].fc
        else:
            old_classes=model.module.classifier._modules[model.last_layer_index].out_features
            pre_out_layer=model.module.classifier._modules[model.last_layer_index]
        #----initialize the new layer------#
        new_output_layer = nn.Linear(model.num_ftrs, old_classes + dset_classes,bias=bias).to(cuda_device)
        #----create a randomly initialized layer for the new classes---#
        new_task_output_layer = nn.Linear(model.num_ftrs, dset_classes,bias=bias).to(cuda_device)
        #----copying the weight----#
        new_weight=torch.cat((pre_out_layer.weight.detach().clone(),new_task_output_layer.weight.detach().clone()),dim=0)
        new_output_layer.weight.data.copy_(new_weight)
        if  bias:
            #-----copying the bias----#
            new_bias=torch.cat((pre_out_layer.bias.detach().clone(),new_task_output_layer.bias.detach().clone()),dim=0)
            new_output_layer.bias.data.copy_(new_bias)
        #----------
        if isinstance(model.module, ResNet18_imprint):
            model.module.classifier._modules[model.last_layer_index].fc=new_output_layer
        else:
            model.module.classifier._modules[model.last_layer_index] = new_output_layer
    else:
        #model.last_layer_index this should be 0
        if  isinstance(model.module,ResNet18_imprint):
            model.module.classifier._modules[model.last_layer_index].fc = nn.Linear(model.num_ftrs, dset_classes,
                                                                                 bias=bias).to(cuda_device)
        else:
            model.module.classifier._modules[model.last_layer_index] = nn.Linear(model.num_ftrs, dset_classes,bias=bias).to(cuda_device)
    return model
