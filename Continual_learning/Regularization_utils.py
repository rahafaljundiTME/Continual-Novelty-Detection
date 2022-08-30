import torch
from collections import namedtuple
from Continual_learning.Continual_methods import LwF
from Continual_learning.Continual_methods import MAS
from Continual_learning.Continual_methods import SSIL
import Continual_learning.Cl_Models as Cl_Models
"""
Utils functions for CL methods
"""

def prepare_model(model, n_classes, regularization_method, task_index,opt):
    """
    prepare the model for the continual learning process
    """
    #task_index is 1 based
    if not hasattr(opt,"shared_head"):
        opt.shared_head=False
    if not hasattr(model,'last_layer_index'):
        last_layers=[x for x in model.classifier._modules.keys()]
        model.last_layer_index = last_layers[0]
        model.num_ftrs = model.classifier._modules[model.last_layer_index].in_features
    # only applies after learning the first ask
    model, original_model = Cl_Models.prepare_cl_model(model, n_classes, task_index,opt)

    if regularization_method == "MAS":
        original_model = None


    return model, original_model


def model_post_process(model, dataset_path, regularization_method,opt):
    """
    estimate omega for importance weight based methods
    """
    if regularization_method == "MAS":
        MAS.estimate_omega(dataset_path, model,opt=opt)
    return model


def generate_reg_loss(trainer,model, original_model, outputs, labels,inputs, criterion,pre_batch, regularization_method,task_index,pre_classes, opt=namedtuple("device", "cuda")):
    """
    return the corresponding loss depending on which regularization method is used
    """

    total_loss = 0
    tasks_outputs = outputs  # variable rename only
    _, preds = torch.max(tasks_outputs[-1].data, 1)
    if regularization_method =="LwF" or regularization_method=="SSIL":
        if original_model is not None:

            original_model.zero_grad()
            original_model.train()
            original_model=original_model.to(opt.device)
    if pre_batch is not None:
        pre_input, pre_label = pre_batch
        pre_outputs = model(pre_input)[-1]  # shared head
        if regularization_method == "SSIL":
            total_loss,task_loss=SSIL.loss(original_model,task_index,pre_input,tasks_outputs,labels,pre_outputs,pre_label,pre_classes,trainer.pre_dset_classes)
            return total_loss, task_loss, preds#=====================
        else:
            #ER
            pre_loss = criterion(pre_outputs, pre_label)
            total_loss += pre_loss

    task_loss = criterion(tasks_outputs[-1], labels)

    total_loss += task_loss

    if regularization_method == "LwF":
        temperature = 2
        if original_model is not None:
            # forward
            # tasks_outputs and target_logits are lists of outputs for each task in the previous model and current model
            with torch.no_grad():
                target_logits = original_model(inputs)


            scale = [item.size(-1) for item in target_logits]
            # Compute distillation loss.
            dist_loss = 0
            # Apply distillation loss to all old tasks.

            for idx in range(len(target_logits)):
                dist_loss += LwF.Fdistillation_loss(tasks_outputs[idx], target_logits[idx], temperature, scale[idx])
            # backward + optimize only if in training phase

        else:

            dist_loss=0

        total_loss += dist_loss

    if regularization_method == "MAS":

        # backward + optimize only if in training phase


        # check!!!!!!!!!!!! also if loss is changing value.
        # add loss from
        if hasattr(model, 'reg_params'):

            for name, param in model.named_parameters():
                reg_param = model.reg_params.get(param)
                if (reg_param is not None) :

                    total_loss += model.reg_lambda / 2. * torch.sum(
                        reg_param['omega'] * torch.pow((param - reg_param['init_val']),2))


    return total_loss, task_loss,preds