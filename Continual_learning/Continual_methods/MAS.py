# Implements the main functions for using MAS regularizer
from __future__ import print_function, division

import torch
from collections import namedtuple

from Continual_learning.Cl_Models import Multihead_Model

"""
functions to handle omega operations
"""
def initialize_reg_params(model,opt):
    reg_params = {}
    for name, param in model.named_parameters():
        if not(isinstance(model,Multihead_Model )) or not("classifier" in name) : #In case of multhead the classifier reg-params are not computed!
            print('initializing param', name)
            omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param = {}
            reg_param['omega'] = omega
            # initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param
    return reg_params


# set omega to zero but after storing its value in a temp omega in which later we can accumolate them both
def initialize_store_reg_params(model,opt):
    reg_params = model.reg_params
    for name, param in model.named_parameters():

        if param in reg_params:
            reg_param = reg_params.get(param)
            print('storing previous omega', name)
            prev_omega = reg_param.get('omega')
            new_omega = torch.FloatTensor(param.size()).zero_()
            init_val = param.data.clone()
            reg_param['prev_omega'] = prev_omega
            reg_param['omega'] = new_omega
            #check if previous omega is different from new omega

            # initialize the initial value to that before starting training
            reg_param['init_val'] = init_val
            reg_params[param] = reg_param


    return reg_params

def accumelate_reg_params(model,opt):
    reg_params = model.reg_params
    for name, param in model.named_parameters():

        if param in reg_params:
            reg_param = reg_params.get(param)
            print('restoring previous omega', name)
            prev_omega = reg_param.get('prev_omega')
            prev_omega = prev_omega.to(opt.device)

            new_omega = (reg_param.get('omega')).to(opt.device)
            acc_omega = torch.add(prev_omega, new_omega)

            del reg_param['prev_omega']
            reg_param['omega'] = acc_omega

            reg_params[param] = reg_param
            del acc_omega
            del new_omega
            del prev_omega

    return reg_params
"""
end of functions to handle omega operations
"""

def estimate_current_importance_weight(model, batch_index, batch_size,opt):
    reg_params = model.reg_params
    for name, param in model.named_parameters():
        reg_param = reg_params.get(param)

        if reg_param is not None and param.grad is not None:# last output layer if head is not shared will have non reg param
            unreg_dp = param.grad.data.clone()

            zero = torch.FloatTensor(param.data.size()).zero_()
            omega = reg_param.get('omega')
            omega = omega.to(opt.device)

            # sum up the magnitude of the gradient
            prev_size = batch_index * batch_size
            curr_size = (batch_index + 1) * batch_size
            omega = omega.mul(prev_size)

            omega = omega.add(unreg_dp.abs_())
            omega = omega.div(curr_size)
            if omega.equal(zero.to(opt.device)):
                print('omega after zero')

            reg_param['omega'] = omega
            # pdb.set_trace()
            reg_params[param] = reg_param


def compute_importance_l2(model, dset_loaders, opt=namedtuple("device", "cuda")):

    model.eval()  # Set model to training mode so we get the gradient

    # Iterate over data.
    index = 0
    for dset_loader in dset_loaders.values():
        for inputs, labels in dset_loader:

            #
            inputs, labels = inputs.to(opt.device),labels.to(opt.device)

            # zero the parameter gradients
            model.zero_grad()

            # forward
            outputs = model(inputs)
            if isinstance(outputs,list):
                outputs=outputs[-1]
            MSE = torch.nn.functional.mse_loss(outputs, torch.zeros(outputs.size()).to(opt.device))
            MSE.backward()
            # print('step')
            estimate_current_importance_weight(model, index, 1,opt=opt) #gradients are already normalized by batch size!

            index += 1

    return model



def replace_heads(previous_model_path, current_model_path):
    current_model = torch.load(current_model_path)

    previous_model = torch.load(previous_model_path)
    current_model.classifier._modules['6'] = previous_model.classifier._modules['6']
    return current_model

def estimate_omega(dataset_path, model, batch_size=200,  opt={}):
    """
    to be called at the end of the training.
    """
    b1 = opt.b1

    if b1:
        # compute the importance with batch size of 1
        update_batch_size = 1
    else:
        update_batch_size = batch_size

    dsets = torch.load(dataset_path)
    # dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=update_batch_size,
    #                                                shuffle=True, num_workers=4)
    #                 for x in ['train', 'val']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=update_batch_size,
                                                   shuffle=True, num_workers=4)
                    for x in ['train']}

    if not hasattr(model, 'reg_params'):
        reg_params = initialize_reg_params(model,opt=opt)
        model.reg_params = reg_params

    reg_params = initialize_store_reg_params(model,opt=opt)
    model.reg_params = reg_params




    print('********************objective with L2 norm***************')
    model = compute_importance_l2(model, dset_loaders, opt)

    sanitycheck(model)
    reg_params = accumelate_reg_params(model,opt=opt)
    model.reg_params = reg_params
    sanitycheck(model)
    return model












def sanitycheck(model):
    for name, param in model.named_parameters():
        # w=torch.FloatTensor(param.size()).zero_()
        print(name)
        if param in model.reg_params:
            reg_param = model.reg_params.get(param)
            omega = reg_param.get('omega')

            print('omega max is', omega.max().item())
            print('omega min is', omega.min().item())
            print('omega mean is', omega.mean().item())






