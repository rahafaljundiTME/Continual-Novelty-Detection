import torch
import time
import numpy as np
import torch.nn as nn
from  NoveltyDetectionModules.NvDModules import NvDModule
import pdb
import numpy as np
import copy
from NoveltyDetectionModules import VAE


class Args(object):

    pass


class Gen_based(NvDModule):

    """
    """
    def __init__(self,model,opts):
        self.CUDA_DEVICE=opts.device
        self.criterion=nn.CrossEntropyLoss() if not hasattr(opts, "novelty_criterion") else opts.criterion
        self.noiseMagnitude=opts.novelty_magnitude
        self.temper=opts.novelty_temperature
        self.end = 1#double number of classes in cifar
        self.warmup=opts.novelty_warmup
        self.max_beta=opts.novelty_max_beta
        self.n_epochs=opts.novelty_n_epochs
        self.batch_size=opts.novelty_batchsize if  hasattr(opts, "novelty_batchsize") else 200
        self.feature_based=opts.novelty_feature_based if  hasattr(opts, "novelty_feature_based") else False
        super(Gen_based, self).__init__(model,opts)
        #=======create arguments for the VAE
        args=Args()
        args.z_size=opts.novelty_z_size
        args.input_size=self.input_size #
        args.gen_hiddens=args.z_size*2
        if not self.feature_based:
            args.input_type="continuous"
        else:
            args.input_type = "binary"
            args.input_size=[model.num_ftrs]#check!
            args.gen_architecture="MLP"
        args.CUDA_DEVICE=self.CUDA_DEVICE
        self.gen = VAE.VAE(args).to(self.CUDA_DEVICE)
        #================================

    def get_data_prob(self, data_set, task_index):

        predicted_probs=[]

        self.gen.eval()  #
        dset_loader = torch.utils.data.DataLoader(data_set, 1,
                                                  shuffle=False)

        self.model.eval()
        self.model.module.eval()
        for j, data in enumerate(dset_loader):
            images, _ = data
            inputs = images.cuda(self.CUDA_DEVICE)
            with torch.no_grad():
                if self.feature_based:
                    features=self.model.module.return_hidden(inputs)#train on the features before the classification layer
                    inputs=features
                x_mean, z_mu, z_var, ldj, z0, zk = self.gen(inputs)

                gen_loss, rec, kl, _ = VAE.calculate_loss(x_mean, inputs, z_mu,
                                                          z_var, z0, zk, ldj, beta=self.beta,reduction="sum")# this is the last beta set during training!
                #prob=1-nn.functional.sigmoid(rec)
                prob=1/(rec+kl) #nn.functional.sigmoid(1/rec)
                #prob=1-nn.functional.sigmoid(gen_loss)
            predicted_probs.append(prob.cpu().numpy())
            #print("sample numner:",j)
            predicted_probs.append(prob.numpy())
        return predicted_probs


    def train(self, dset_in, dset_out, task_index,train_in):
        #if the training is called twice for a new task then it will continue training


        dset_in_loaders = {x: torch.utils.data.DataLoader(dset_in[x], self.batch_size,
                                                       shuffle=True, num_workers=4)
                        for x in ['train', 'val', 'test']}
        # if train_in is not None:
        #     dset_in_loaders["train"]=torch.utils.data.DataLoader(train_in, self.batch_size,
        #                                                shuffle=True, num_workers=4)
        self.gen.train()  #
        opt_gen = torch.optim.Adam(self.gen.parameters())#optimizer is renewed
        sample_amt = 0
        self.model.eval()
        # ----------------
        # Begin Epoch Loop
        best_val_loss=10**8
        for epoch in range(self.n_epochs):
            self.gen.train()
            #---------------------
            # Begin Minibatch Loop
            for i, (data, target) in enumerate(dset_in_loaders["train"]):


                #if args.cuda:
                data, target = data.to(self.CUDA_DEVICE), target.to(self.CUDA_DEVICE)
                sample_amt += data.size(0)
                self.beta = min([(sample_amt) / max([self.warmup, 1.]), self.max_beta])

                #------ Train Generator ------#

                #-------------------------------
                # Begin Generator Iteration Loop
                if self.feature_based:
                    with torch.no_grad():
                        features=self.model.module.return_hidden(data)#train on the features before the classification layer
                        data=features
                #============
                x_mean, z_mu, z_var, ldj, z0, zk = self.gen(data)
                gen_loss, rec, kl, _ = VAE.calculate_loss(x_mean, data, z_mu,
                        z_var, z0, zk, ldj, beta=self.beta,reduction="sum")#before submission it was sum
                #====================



                opt_gen.zero_grad()
                gen_loss.backward()
                opt_gen.step()

                # End Generator Iteration Loop
                #------------------------------

                print('current VAE loss = {:.4f} (rec: {:.4f} + beta: {:.2f} * kl: {:.2f})'
                            .format(gen_loss.item(), rec.item(), self.beta, kl.item()))

            #EVALUATION
            self.gen.eval()  #
            total_loss=0
            for i, (data, target) in enumerate(dset_in_loaders["val"]):


                #if args.cuda:
                with torch.no_grad():
                    data, target = data.to(self.CUDA_DEVICE), target.to(self.CUDA_DEVICE)
                    sample_amt += data.size(0)
                    self.beta = min([(sample_amt) / max([self.warmup, 1.]), self.max_beta])

                    #------ Train Generator ------#

                    #-------------------------------
                    # Begin Generator Iteration Loop
                    if self.feature_based:
                        with torch.no_grad():
                            features = self.model.module.return_hidden(
                                data)  # train on the features before the classification layer
                            data = features
                    x_mean, z_mu, z_var, ldj, z0, zk = self.gen(data)
                    gen_loss, rec, kl, _ = VAE.calculate_loss(x_mean, data, z_mu,
                            z_var, z0, zk, ldj, beta=self.beta)

                    total_loss+=gen_loss
            total_loss=total_loss / i
            print("This eval loss is ",total_loss )
            if total_loss<best_val_loss:
                print("getting better val loss")
                best_val_loss=total_loss
                best_gen=copy.deepcopy(self.gen)

        self.gen=best_gen
        return