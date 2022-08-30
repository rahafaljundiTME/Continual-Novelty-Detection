import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import numpy as np
from NoveltyDetectionModules.NvDModules import NvDModule

from Continual_learning.Architecture import VGGSlim2, ResNet, ResNet_dropout, VGGSlim2_dropout, alex_seq
"""
A fake CND model, to extract the stats of the features of the different sets at the different stages of the sequence. 
"""

class Open_Stat(NvDModule):

    def __init__(self, model, opts):
        self.CUDA_DEVICE = opts.device
        self.name = ""
        self.criterion = nn.CrossEntropyLoss() if not hasattr(opts, "novelty_criterion") else opts.criterion
        self.noiseMagnitude = opts.novelty_magnitude
        # ================================stats==============================
        self.features_norm_bef = []
        self.features_norm_aft = []
        self.cos_softmax = []  # biased_prod_max
        self.used_delta = []  # biased_prod_std
        self.prod_max = []  # prod_max
        self.prod_std = []  # prod_std
        self.org_sotmax = []  # prod_aft_max
        self.delta = []
        self.mean_dot_product = []
        self.features_norm=[]
        self.features_norm_std=[]
        # self.names=["in_val","out_val","1in","1out","2out","3out"]
        self.get_layers_names(model)
        super(Open_Stat, self).__init__(model, opts)

    def train(self, dset_in, dset_out, task_index, train_in):
        self.hooks = {}
        if 1:
            for name, module in self.model.named_modules():
                if name in self.layers_names:
                    self.hooks[name] = module.register_forward_hook(self.extract_layer_output)

    def extract_layer_output(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        # print('Inside ' + self.__class__.__name__ + ' forward')
        #
        # self.features.append(output)
        self.features_norm.append(torch.norm(input[0]).detach().cpu().numpy())
        self.features_norm_std.append(np.std(input[0].detach().cpu().numpy()))

    def get_data_prob(self, data_set, task_index):
        softmax = torch.nn.Softmax(dim=1)
        # import pdb;pdb.set_trace()
        # ============================stats================
        while len(self.features_norm_bef) <= task_index:
            # ====new task
            self.features_norm_bef.append({})
            self.features_norm_aft.append({})
            self.cos_softmax.append({})
            self.used_delta.append({})
            self.prod_max.append({})
            self.prod_std.append({})
            self.org_sotmax.append({})
            self.delta.append({})
            self.mean_dot_product.append({})
        self.features_norm_bef[task_index][self.name] = []
        self.features_norm_aft[task_index][self.name] = []
        self.cos_softmax[task_index][self.name] = []
        self.used_delta[task_index][self.name] = []
        self.prod_max[task_index][self.name] = []
        self.prod_std[task_index][self.name] = []
        self.org_sotmax[task_index][self.name] = []
        self.delta[task_index][self.name] = []
        self.mean_dot_product[task_index][self.name] = []
        # =============================================================
        # def testData(model, CUDA_DEVICE, dataloader):
        if self.opts.shared_head:
            head_index = 0
        else:
            head_index = task_index
        predicted_probs = []
        dset_loader = torch.utils.data.DataLoader(data_set, 1,
                                                  shuffle=False)
        self.model.eval()
        if hasattr(self.model.module, "embedding"):
            # IMPRINTING
            self.model.module.embedding.normalize = False

            final_layer = self.model.module.classifier._modules[str(head_index)].fc
        else:
            final_layer = self.model.module.classifier._modules[str(head_index)]  # check which type the model is off!

        if hasattr(final_layer, "bias") and not final_layer.bias is None:
            biased_final_layer_weight = torch.cat((final_layer.weight, final_layer.bias.unsqueeze(1).detach()), dim=1)
            self.bias = True
        else:
            self.bias = False

        # normalize the weights
        normed_weight = F.normalize(final_layer.weight, dim=1, p=2)  #
        if self.bias:
            biased_normed_weight = F.normalize(biased_final_layer_weight, dim=1, p=2)  #
        final_layer_weight = normed_weight.detach()

        class_sim = torch.zeros(final_layer_weight.size(0), final_layer_weight.size(0))

        if 0:
            for i in range(final_layer_weight.size(0)):
                # torch.cat((final_layer_weight[0:i], final_layer_weight[i + 1:]), dim=0)
                class_sim[i] = F.linear(final_layer_weight[i], final_layer_weight)

        # samples_sim = torch.zeros(final_layer_weight.size(0), final_layer_weight.size(0))
        # sigmoid = torch.nn.Sigmoid()
        acc = 0
        if 0:
            class_sim_inds = torch.zeros(final_layer_weight.size(0), final_layer_weight.size(0))
            # import pdb;pdb.set_trace()
            classes_count = []  # [0 for i in range(final_layer_weight.size(0))]
            for i in range(final_layer_weight.size(0)):
                v, class_sim_inds[i] = torch.sort(class_sim[i], descending=True)
                classes_count.append(class_sim_inds[i][-1].cpu().numpy())  # it was 1
            class_sim_inds = class_sim_inds.long().to(self.CUDA_DEVICE)
        # import pdb;pdb.set_trace()
        for j, data in enumerate(dset_loader):
            #

            images, labels = data
            inputs = images.cuda(self.CUDA_DEVICE)
            labels = labels.cuda(self.CUDA_DEVICE)
            if 0:
                ###########extraction of all layers features #######

                self.features_norm = []
                self.features_norm_std = []
                with torch.no_grad():
                    self.model(inputs)
                    self.biased_prod_max[task_index][self.name].append(sum(self.features_norm))
                    self.biased_prod_std[task_index][self.name].append(sum(self.features_norm_std))
                #################all layers features #######

            self.model.zero_grad()
            if 1:
                with torch.no_grad():
                    # using softmax plus output
                    full_output = self.model(inputs)[head_index]
                    softmaxoutput, prediction = torch.max(softmax(full_output), dim=1)
                    self.org_sotmax[task_index][self.name].append(softmaxoutput.detach().cpu().numpy())

            features = self.model.module.return_hidden(inputs)
            features.retain_grad()
            features_normed = F.normalize(features, p=2, dim=1)
            with torch.no_grad():
                pre_featnorm = torch.norm(features)
            self.features_norm_bef[task_index][self.name].append(pre_featnorm.detach().cpu().numpy())

            if self.bias:
                with torch.no_grad():
                    biased_features_norm = F.normalize(
                        torch.cat((features, torch.ones(features.size(0), 1).to(self.CUDA_DEVICE)), dim=1), p=2, dim=1)
            if hasattr(self.model.module, "embedding"):
                final_layer = self.model.module.classifier._modules[
                    str(head_index)].fc  # check which type the model is off!
            else:
                final_layer = self.model.module.classifier._modules[
                    str(head_index)]  # check which type the model is off!
            # normalize the weights
            normed_weight = F.normalize(final_layer.weight, dim=1, p=2)  #
            dotproduct = F.linear(features_normed, normed_weight)
            bef_dot_product = dotproduct.clone()
            # *************************************
            self.prod_max[task_index][self.name].append(torch.max(bef_dot_product).detach().cpu().numpy())

            self.prod_std[task_index][self.name].append(np.std(bef_dot_product.detach().cpu().numpy()))

            # *****************************************
            if self.bias:
                biased_dot_product = F.linear(biased_features_norm, biased_final_layer_weight)

            output, prediction = torch.max(softmax(dotproduct * pre_featnorm), dim=1)  # --------------------new.
            bef_out = output.clone()
            # *****************
            self.cos_softmax[task_index][self.name].append(bef_out.detach().cpu().numpy())
            exc_dot_product=torch.cat((bef_dot_product[0:prediction],bef_dot_product[prediction+1:]),dim=0)
            mean_dot_prodocut=np.mean(exc_dot_product.detach().cpu().numpy())
            self.mean_dot_product[task_index][self.name].append(mean_dot_prodocut)
            if self.bias:
                biased_output, biased_prediction = torch.max(biased_dot_product, 1)
                self.biased_prod_max[task_index][self.name].append(biased_output.detach().cpu().numpy())
                self.biased_prod_std[task_index][self.name].append(np.std(biased_dot_product.detach().cpu().numpy()))
            # *****************

            predicted_label = prediction.clone()

            v, sample_sim_inds = torch.sort(bef_dot_product, descending=True)
            # import pdb;pdb.set_trace()
            this_label = torch.tensor(sample_sim_inds[0].detach().cpu().numpy()[1]).long().to(
                self.CUDA_DEVICE).unsqueeze(0)  # the second similar class
            # this_label=torch.tensor(classes_count[predicted_label]).long().to(self.CUDA_DEVICE).unsqueeze(0)
            loss = self.criterion(dotproduct, this_label)
            loss.backward()

            with torch.no_grad():

                features = torch.add(features, features.grad,
                                     alpha=-self.noiseMagnitude)  # the gradient should then be computed to the noralized features!
                featnorm = torch.norm(features)
                self.features_norm_aft[task_index][self.name].append(featnorm.detach().cpu().numpy())
                features = F.normalize(features, p=2, dim=1)

                dotproduct = F.linear(features, final_layer_weight)
                output, prediction = torch.max(softmax(dotproduct), dim=1)  # ---newtorch.max(dotproduct, 1)
                prod_aft_max = output.clone()
                # *****************************

                # **********************************

                this_delta = torch.norm(dotproduct[0] - bef_dot_product[0]).cpu()
                # ***********************
                self.delta[task_index][self.name].append(this_delta.detach().cpu().numpy())

                # print(j, samples_sim[prediction])
                # pre_featnorm=0.5 if pre_featnorm>20 else pre_featnorm/50
                delta = dotproduct[0][predicted_label] - dotproduct[0][this_label]
                self.used_delta[task_index][self.name].append(delta.detach().cpu().numpy())
                #      dotproduct[0][predicted_label],
                #      dotproduct[0][this_label])
                # prob=dotproduct[0][predicted_label]+(bef_dot_product[0][predicted_label] - bef_dot_product[0][this_label])-(                                                                                                                                                                     dotproduct[0][predicted_label] - dotproduct[0][this_label])
                if self.z_size == 1:
                    prob = bef_out + delta
                if self.z_size==2:
                    prob = bef_out - mean_dot_prodocut
                if self.z_size==3:
                    prob = bef_out - mean_dot_prodocut+ delta
                # prob = bef_out + this_delta
                if self.z_size==4:
                    prob=pre_featnorm#softmaxoutput+ delta
                if self.z_size==5:
                    prob=torch.sigmoid(10* delta)        
                if self.z_size==6:
                    #import pdb;pdb.set_trace()
                    prob=torch.softmax(torch.tensor([dotproduct[0][predicted_label], dotproduct[0][this_label]])*pre_featnorm.cpu(),dim=0)[0]
                predicted_probs.append(prob.cpu().numpy())

                acc += sum(prediction == labels)

                del inputs, images, data

                torch.cuda.empty_cache()
        del dset_loader

        print("acc :", int(acc) * 100 / (len(data_set)))
        # import pandas as pd; import seaborn as sn; import matplotlib.pyplot as plt;
        # plt.clf()
        # samples_sim=samples_sim.numpy()
        # df_cm = pd.DataFrame(samples_sim) ; svm = sn.heatmap(df_cm,cmap='coolwarm', linecolor='white', linewidths=1)
        # plt.savefig('confusion/confusion'+self.name+'.png', dpi=400)

        # self.name=self.names[self.names.index(self.name)+1]
        torch.cuda.empty_cache()
        # import pdb;pdb.set_trace()
        return predicted_probs  # ,samples_sim

    def get_layers_names(self, model):
        # to update the name for ResNet!
        layers_names = []
        if isinstance(model.module, VGGSlim2) or isinstance(model.module, VGGSlim2_dropout):
            layers_names = ["module.features.11", "module.features.13", "module.avgpool", "module.fc.0", "module.fc.3",
                            "module.classifier.0"]
        if isinstance(model.module, ResNet):
            layers_names = ["module.layer2.1.bn2", "module.layer3.1.bn2", "module.layer4.0.bn2", "module.layer4.1.bn2"]
        if isinstance(model.module, ResNet_dropout):
            layers_names = ["module.dropout", "module.layer4.0.conv1", "module.layer3.0.conv1", "module.layer2.0.conv1",
                            "module.layer1.0.conv1"]
        if isinstance(model.module, alex_seq):
            layers_names = ["module.features.8", "module.features.10", "module.fc.1", "module.fc.4",
                            "module.classifier.0"]
        self.layers_names = layers_names

    def exit_protocols(self):
        if 1:
            for hook in self.hooks.values():
                hook.remove()