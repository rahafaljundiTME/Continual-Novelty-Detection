import torch
import time
import numpy as np
import sklearn.covariance
import torch.nn as nn
from  NoveltyDetectionModules.NvDModules import NvDModule
from sklearn.linear_model import LogisticRegressionCV
from NoveltyDetectionModules.Metrics   import *
from Continual_learning.Architecture import  VGGSlim2,ResNet,ResNet_dropout,VGGSlim2_dropout,alex_seq
# For evaluation, we use a thresholdbased detector which measures some confidence score of the test sample, and then classifies the
# test sample as in-distribution if the confidence score is above some threshold.
class Mahalanobis(NvDModule):
    """
    """
    def __init__(self,model,opts):
        self.CUDA_DEVICE=opts.device
        self.criterion=nn.CrossEntropyLoss() if not hasattr(opts, "novelty_criterion") else opts.criterion
        self.batch_size=opts.novelty_batchsize if  hasattr(opts, "novelty_batchsize") else 200
        self.layer_index=opts.novelty_layer_index
        self.magnitude=opts.novelty_magnitude
        self.end = 1
        self.get_layers_names(model)
        self.num_classes=0
        self.dist=[] #initialization of the sample to class mean distance for each features
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        super(Mahalanobis, self).__init__(model, opts)

    def get_data_prob(self, data_set, task_index):
        """
        gets the predicted probailities of in distribution for the given set
        :param data_set:
        :param task_index:
        :return:
        """
        dset_loader = torch.utils.data.DataLoader(data_set, self.batch_size,
                                                       shuffle=False)
        print('get Mahalanobis scores')
        #magenitude m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005] #to be used for tunning the hyper parameters

        #for i in range(self.num_output): also ti
        for layer_index in range(len(self.layers_names)):
            self.layer_index=layer_index# this is not a parameter any more!
            M_sc = self.get_Mahalanobis_score(dset_loader, task_index)
            M_sc = np.asarray(M_sc, dtype=np.float32)

            if layer_index==0:
                M_all_sc=M_sc.reshape((M_sc.shape[0], -1))
            else:
                M_all_sc = np.concatenate((M_all_sc, M_sc.reshape((M_sc.shape[0], -1))), axis=1)

        sc_y_pred = self.lr.predict_proba(M_all_sc)[:, 1]#check 1 or 0?!
        #try returning the average?
        M_avg_sc=M_all_sc.mean(axis=1)
        return sc_y_pred

    def train(self,dset_in,dset_out,task_index,train_in):
        #dsetout is used to train the logistic regression model and tune hyperparameters mainly the magnitude of added noise
        in_num_classes = len(dset_in["test"].classes)
        out_num_classes = len(dset_out["test"].classes)
        dset_in_loaders = {x: torch.utils.data.DataLoader(dset_in[x], self.batch_size,
                                                       shuffle=False, num_workers=4)
                        for x in ['train', 'val', 'test']}
        dset_out_loaders = {x: torch.utils.data.DataLoader(dset_out[x], self.batch_size,
                                                       shuffle=False, num_workers=4)
                        for x in ['train', 'val', 'test']}

        self.model.eval()
        if train_in is not None:
            dset_in_loaders["train"]=torch.utils.data.DataLoader(train_in, self.batch_size,
                                                       shuffle=False, num_workers=4)
        #estimate the mean and precision matrices
        self.estimate_mean_covariance(dset_in_loaders,num_classes=in_num_classes, task_index=task_index)

        #train the detection
        print('get Mahalanobis scores')
        # for i in range(self.num_output): also ti
        for layer_index in range(len(self.layers_names)):
            self.layer_index = layer_index
            M_in_train = self.get_Mahalanobis_score(dset_in_loaders["train"], task_index)
            M_in_train = np.asarray(M_in_train, dtype=np.float32)
            M_in_val = self.get_Mahalanobis_score(dset_in_loaders["val"], task_index)
            M_in_val = np.asarray(M_in_val, dtype=np.float32)
            M_out_train = self.get_Mahalanobis_score(dset_out_loaders["train"], task_index)
            M_out_val = self.get_Mahalanobis_score(dset_out_loaders["val"], task_index)
            M_out_train = np.asarray(M_out_train, dtype=np.float32)
            M_out_val = np.asarray(M_out_val, dtype=np.float32)
            if layer_index==0:
                M_all_in_train=M_in_train.reshape((M_in_train.shape[0], -1))
                M_all_in_val = M_in_val.reshape((M_in_val.shape[0], -1))
                M_all_out_train = M_out_train.reshape((M_out_train.shape[0], -1))
                M_all_out_val = M_out_val.reshape((M_out_val.shape[0], -1))
            else:

                M_all_in_train = np.concatenate((M_all_in_train, M_in_train.reshape((M_in_train.shape[0], -1))), axis=1)
                M_all_in_val = np.concatenate((M_all_in_val, M_in_val.reshape((M_in_val.shape[0], -1))), axis=1)
                M_all_out_train = np.concatenate((M_all_out_train, M_out_train.reshape((M_out_train.shape[0], -1))), axis=1)
                M_all_out_val = np.concatenate((M_all_out_val, M_out_val.reshape((M_out_val.shape[0], -1))), axis=1)


        Mahalanobis_train_data, Mahalanobis_train_labels=self.merge_and_generate_labels(M_all_in_train,M_all_out_train)
        #Mahalanobis_val_data, Mahalanobis_val_labels = self.merge_and_generate_labels(M_in_val, M_out_val)
        lr = LogisticRegressionCV(n_jobs=-1).fit(Mahalanobis_train_data, Mahalanobis_train_labels)# this is only trained on the newest dataset

        in_y_pred=lr.predict_proba(M_all_in_val)[:, 1]

        out_y_pred = lr.predict_proba(M_all_out_val)[:, 1]

        ## print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
        #y_pred = lr.predict_proba(X_val_for_test)[:, 1]
        fpr = tpr95(in_y_pred, out_y_pred, in_num_classes,self.end)
        self.lr=lr

        return fpr

    def estimate_mean_covariance(self, dset_loaders, num_classes, task_index):
         #first time, no continual learning
        self.hooks = {}
        self.features=[]

        for name, module in self.model.named_modules():
            if name in self.layers_names:
                self.hooks[name] = module.register_forward_hook(self.extract_layer_output)
        # set information about feature extaction
        self.model.eval()
        self.model.module.eval()
        temp_x = torch.rand(self.input_size).unsqueeze(0).to(self.CUDA_DEVICE)
        self.model(temp_x)
        num_output = len(self.features)
        self.num_output=num_output
        self.feature_list = np.empty(num_output)
        count = 0
        for out in self.features:
            self.feature_list[count] = out.size(1)
            count += 1
        del temp_x
        torch.cuda.empty_cache()
        self.sample_estimator( num_classes, dset_loaders["train"],task_index)

        return

    def extract_layer_output(self,module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        #print('Inside ' + self.__class__.__name__ + ' forward')
        #
        #self.features.append(output)
        self.features.append(input[0])


    def sample_estimator(self, num_classes, train_loader,task_index):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """

        #remember to remove the hook, empty the outputs
        if self.opts.shared_head:
            head_index=-1
        else:
            head_index=task_index
        self.model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(self.feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        for data, target in train_loader:
            total += data.size(0)
            data = data.to(self.CUDA_DEVICE)
            self.features=[]
            output = self.model(data)
            if isinstance(output, list):
                output = output[head_index]
            out_features= self.features.copy()
            del self.features
            torch.cuda.empty_cache()
            self.features = []
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)

            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            del output
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if label>=num_classes:
                    #shared head with Train-in as set
                    label=label-task_index*num_classes #assuming in shared head each tak number of classes is equal
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                            = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1
                num_sample_per_class[label] += 1
            del data, target
            torch.cuda.empty_cache()
        sample_class_mean = []
        out_count = 0
        for num_feature in self.feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
            #del list_features[k]
            # find inverse
            #
            if len(self.dist)<num_output:
                #continual learning, not the first task
                self.dist.append( X.cpu())

            else:

                self.dist[k] = torch.cat((self.dist[k] , X.cpu()), 0)

            group_lasso.fit(self.dist[k].numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float()#.cuda()
            precision.append(temp_precision)

        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
        if not hasattr(self,"sample_mean"):
            self.sample_mean=sample_class_mean
        else:

            for k in range(num_output):
                self.sample_mean[k]=torch.cat((self.sample_mean[k],sample_class_mean[k]),0) #update class mean in case of continual learning
        self.num_classes+=num_classes
        del list_features
        del X
        torch.cuda.empty_cache()
        self.precision=[]
        for k in range(num_output):
            self.precision.append( precision[k].cuda())

    def get_Mahalanobis_score(self, test_loader,task_index):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''
        #     layer_index, magnitude
        if self.opts.shared_head:
            head_index=-1
        else:
            head_index=task_index
        self.model.eval()
        Mahalanobis_dist = []


        for data, target in test_loader:#batch or one sample
            self.model.zero_grad()
            data=data.to(self.CUDA_DEVICE)

            data.requires_grad=True
            self.features=[]
            output = self.model(data) #layer index should be treated in the hook
            if isinstance(output, list):
                output = output[head_index]
            out_features=self.features[self.layer_index]
            self.features=[]#
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)#

            # compute Mahalanobis score
            gaussian_score = 0
            with torch.no_grad():
                for i in range(self.num_classes):
                    batch_sample_mean = self.sample_mean[self.layer_index][i]

                    zero_f = out_features.data - batch_sample_mean
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[self.layer_index]), zero_f.t()).diag()
                    if i == 0:
                        gaussian_score = term_gau.view(-1, 1)
                    else:
                        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = self.sample_mean[self.layer_index].index_select(0, sample_pred)
            zero_f = out_features - batch_sample_mean
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[self.layer_index]), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad, 0)
            gradient = (gradient.float() - 0.5) * 2
            # if net_type == 'densenet':
            #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
            #                          gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
            #                          gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
            #                          gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
            #check on this gradient normalization values for input perturbation
            # gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
            #                      gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))#
            # gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
            #                      gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            # gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
            #                      gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (self.data_std[0]))# These are the std values used for TinyImageNet
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (self.data_std[1]))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (self.data_std[2]))

            with torch.no_grad():
                tempInputs = torch.add(data, gradient,alpha= -self.magnitude)

                output = self.model(tempInputs)
                noise_out_features=self.features[self.layer_index]
                self.features=[]
                noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
                noise_out_features = torch.mean(noise_out_features, 2)
                noise_gaussian_score = 0
                for i in range(self.num_classes):
                    batch_sample_mean = self.sample_mean[self.layer_index][i]
                    zero_f = noise_out_features.data - batch_sample_mean
                    term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[self.layer_index]), zero_f.t()).diag()
                    if i == 0:
                        noise_gaussian_score = term_gau.view(-1, 1)
                    else:
                        noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

                noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
                Mahalanobis_dist.extend(noise_gaussian_score.cpu().numpy())

            # for i in range(data.size(0)):
            #     g.write("{}\n".format(noise_gaussian_score[i]))
        #g.close()

        return Mahalanobis_dist

    # this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
    def merge_and_generate_labels(self,X_pos, X_neg):
        """
        merge positve and nagative artifact and generate labels
        return: X: merged samples, 2D ndarray
                 y: generated labels (0/1): 2D ndarray same size as X
        """
        X_pos = np.asarray(X_pos, dtype=np.float32)
        X_pos = X_pos.reshape((X_pos.shape[0], -1))

        X_neg = np.asarray(X_neg, dtype=np.float32)
        X_neg = X_neg.reshape((X_neg.shape[0], -1))

        X = np.concatenate((X_pos, X_neg))
        y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
        y = y.reshape((X.shape[0], 1))

        return X, y

    def get_layers_names(self,model):
        #to update the name for ResNet!
        layers_names=[]
        if isinstance(model.module,VGGSlim2)or isinstance(model.module,VGGSlim2_dropout):
            layers_names=["module.features.11","module.features.13","module.avgpool","module.fc.0","module.fc.3","module.classifier.0" ]
        if isinstance(model.module,ResNet):
            layers_names = [ "module.layer2.1.bn2", "module.layer3.1.bn2", "module.layer4.0.bn2","module.layer4.1.bn2"]
        if  isinstance(model.module, ResNet_dropout):
            layers_names=["module.dropout","module.layer4.0.conv1","module.layer3.0.conv1","module.layer2.0.conv1","module.layer1.0.conv1"]
        if isinstance(model.module, alex_seq):
            layers_names = ["module.features.8", "module.features.10", "module.fc.1", "module.fc.4",
                            "module.classifier.0"]
        self.layers_names=layers_names

    def exit_protocols(self):
        for hook in self.hooks.values():
            hook.remove()