import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
import copy
from NoveltyDetectionModules.NvDModules import NvDModule

from Continual_learning.Architecture import VGGSlim2, ResNet, ResNet_dropout, VGGSlim2_dropout, alex_seq, \
    ResNet18_imprint

"""
Code for baseline1 (1) and baseline 2 (2)
"""


class Baseline(NvDModule):

    def __init__(self, model, opts):
        self.CUDA_DEVICE = opts.device
        self.name = ""
        self.criterion = nn.CrossEntropyLoss() if not hasattr(opts, "novelty_criterion") else opts.criterion
        self.noiseMagnitude = opts.novelty_magnitude
        self.neighbours = opts.novelty_n_epochs # the number of neighbour classes considered while estimating the criteria

        self.temperature = opts.novelty_temperature


        super(Baseline, self).__init__(model, opts)





    def get_data_prob(self, data_set, task_index):
        softmax = torch.nn.Softmax(dim=1)
        # import pdb;pdb.set_trace()

        if self.opts.shared_head:
            head_index = 0
        else:
            head_index = task_index
        predicted_probs = []
        dset_loader = torch.utils.data.DataLoader(data_set, 1,
                                                  shuffle=False)
        self.model.eval()
        org_model=copy.deepcopy(self.model)

        if hasattr(self.model.module, "embedding"):
            # IMPRINTING
            self.model.module.embedding.normalize = False

            final_layer = self.model.module.classifier._modules[str(head_index)].fc
        else:
            final_layer = self.model.module.classifier._modules[str(head_index)]  # check which type the model is off!

        self.bias = False #it assumes a network with no bias in the last layer.

        # normalize the weights
        normed_weight = F.normalize(final_layer.weight, dim=1, p=2)  #

        final_layer_weight = normed_weight.detach()
        ########################################################

        acc = 0

        # import pdb;pdb.set_trace()
        for j, data in enumerate(dset_loader):
            #
            # import pdb;pdb.set_trace()
            images, labels = data
            inputs = images.cuda(self.CUDA_DEVICE)
            labels = labels.cuda(self.CUDA_DEVICE)

            self.model.zero_grad()

            with torch.no_grad():
                features = self.model.module.return_hidden(inputs)
                not_normed_dotproduct = F.linear(features, final_layer.weight)
                softmax_prob = softmax(not_normed_dotproduct)
                softmaxoutput, softmax_prediction = torch.max(softmax_prob, dim=1)
                pre_featnorm = torch.norm(features)
                #new check
                outputs=org_model(inputs)
                outputs=outputs[head_index]

            features.requires_grad = True

            deltas = []
            for n in range(self.neighbours):
                features_normed = F.normalize(features, p=2, dim=1)
                dotproduct = F.linear(features_normed, final_layer_weight)
                if n == 0:
                    bef_dot_product = dotproduct.clone()

                    output, prediction = torch.max(bef_dot_product, dim=1)  # --------------------new.
                    output = output.detach().cpu()
                    predicted_label = prediction.clone()

                    v, sample_sim_inds = torch.sort(bef_dot_product, descending=True)
                    # import pdb;pdb.set_trace()
                this_label = torch.tensor(sample_sim_inds[0].detach().cpu().numpy()[n + 1]).long().to(
                    self.CUDA_DEVICE).unsqueeze(0)  # the second similar class

                loss = self.criterion(dotproduct, this_label)
                loss.backward()

                with torch.no_grad():

                    updated_features = torch.add(features, features.grad,
                                                 alpha=-self.noiseMagnitude)  # the gradient should then be computed to the noralized features!

                    x_grad = features.grad.clone()
                    updated_features = F.normalize(updated_features, p=2, dim=1)

                    aft_dotproduct = F.linear(updated_features, final_layer_weight)

                    this_delta = (aft_dotproduct[0][predicted_label] - aft_dotproduct[0][this_label]).detach().cpu()
                    deltas.append(this_delta)
                    # import pdb;pdb.set_trace()

                features.grad.zero_()


            delta = sum(deltas) / (n + 1)


            if self.z_size == 1: #Baseline 1
                # import pdb;pdb.set_trace()
                # x_grad = F.normalize(x_grad, p=2, dim=1)
                with torch.no_grad():
                    #softmax_prob = softmax(not_normed_dotproduct)
                    softmax_prob = softmax(outputs)
                    rep_vec = F.linear(softmax_prob, final_layer_weight.transpose(0, 1))
                    # grad_sim=F.normalize(grad_sim,p=2,dim=1)

                    sample_grad_sim = F.linear(rep_vec, features_normed).squeeze().cpu()


                prob = sample_grad_sim


            if self.z_size == 2: #Baseline 2
                with torch.no_grad():
                    softmax_prob = softmax(not_normed_dotproduct)
                    rep_vec = F.linear(softmax_prob, final_layer_weight.transpose(0, 1))
                    # grad_sim=F.normalize(grad_sim,p=2,dim=1)

                    sample_grad_sim = F.linear(rep_vec, features_normed).squeeze().cpu()
                prob = sample_grad_sim + delta

            predicted_probs.append(prob.numpy())
            acc += sum(prediction == labels)

            del inputs, images, data

            torch.cuda.empty_cache()
        del dset_loader

        print("acc :", int(acc) * 100 / (len(data_set)))

        torch.cuda.empty_cache()
        #
        print(self.name, sum(predicted_probs) / len(predicted_probs))
        return predicted_probs  # ,samples_sim



    def exit_protocols(self):
        return
        # if 0:
        #    for hook in self.hooks.values():

