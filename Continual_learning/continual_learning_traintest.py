import torch
import os
import numpy as np
import random
from Continual_learning import Regularized_Training
from Tools.CN_Set import Continual_ConcatDataset
from torch.utils.data.dataset import Subset
from Test_Utils import *
import itertools
"""
Train on a CL sequence with a CL method and test the performance
"""
class Continual_Trainer(object):

    def __init__(self,nb_tasks ,opt,parent_exp_dir,dataset_parent_dir,seed):
        self.nb_tasks=nb_tasks
        self.opt=opt
        self.parent_exp_dir=parent_exp_dir
        self.dataset_parent_dir=dataset_parent_dir
        self.seed=seed
        self.buffer_size=opt.buffer_size
        self.batch_size=opt.batch_size
        self.lr_decay_rate=opt.lr_decay_rate


        super(object, self).__init__()



    def train(self):


        model_path = ''  # ''./vgg11slim2.pth.tar'
        #---replay related----
        pre_dset=None
        self.pre_datasets=[]
        self.pre_inds=[]
        self.pre_dset_classes=[]
        pre_len=0#the length of the previous dataset
        # ---replay related----


        for task in range(1, self.nb_tasks + 1):
            lr=self.opt.lr
            # if task>1:
            #     lr=self.opt.lr*0.1
            # else:
            #     lr=self.opt.lr
            task_name = str(task)
            extra_str=""
            for arg in vars(self.opt):
                extra_str += str(arg ) + '_' + str(getattr(self.opt, arg))

            print(extra_str)



            exp_dir = os.path.join(self.parent_exp_dir, task_name, 'CND/' + 'Seed' + str(self.seed) + extra_str)
            if os.path.isdir(exp_dir):
                continue
            dataset_path = os.path.join(self.dataset_parent_dir, task_name, 'trainval_dataset.pth.tar')
            dsets = torch.load(dataset_path)
            #num_epochs=self.opt.num_epochs
            #batch_size=self.batch_size,
            #regularization_method=self.opt.regularization_method
            # reg_lambda=self.opt.reg_lambda,
            #opt=self.opt
            #lr_decay=self.lr_decay_rate
            Regularized_Training.train_on_task(self,dataset_path=dataset_path,pre_dset=pre_dset, exp_dir=exp_dir,
                                               task_index=task, model_path=model_path,
                                                lr=lr)

            model_path = os.path.join(exp_dir, "best_model.pth.tar")

            if self.buffer_size>0:

                buffer_per_dataset=int(self.buffer_size/task)
                if buffer_per_dataset<1:
                    buffer_per_dataset=1#the allowed minimum
                this_dset_inds = random.sample(range(pre_len,pre_len+len(dsets["train"])),min(len(dsets["train"]), buffer_per_dataset))
                pre_len+=len(dsets["train"])
                for task in range(len(self.pre_inds)):
                    #reduce the number of samples stored from previous datasets
                    self.pre_inds[task]=random.sample(self.pre_inds[task], buffer_per_dataset) #check if it changes
                    self.pre_datasets[task].dataset.use_cache = False #clear cash
                    self.pre_datasets[task].dataset.cached_data = [None for i in range(
                        len(self.pre_datasets[task].dataset.cached_data))]
                    #shift the inds of the datasets!
                self.pre_inds.append(this_dset_inds)
                self.pre_datasets.append(dsets["train"])
                self.pre_dset_classes.append(len(dsets["test"].classes))
                all_pre_datasets=Continual_ConcatDataset(self.pre_datasets,self.pre_dset_classes )

                all_pre_indices=list(itertools.chain.from_iterable( self.pre_inds))#combine all inds
                pre_dset=Subset(all_pre_datasets, all_pre_indices)


class Continual_Tester(object):

    def __init__(self,nb_tasks ,opt,parent_exp_dir,dataset_parent_dir):
        self.nb_tasks=nb_tasks
        self.opt=opt
        self.parent_exp_dir=parent_exp_dir
        self.dataset_parent_dir=dataset_parent_dir

        self.buffer_size=opt.buffer_size
        self.accs = [0 for j in range( self.nb_tasks)]
        self.forgettings = [0 for j in range( self.nb_tasks)]
        self.extra_str = ""
        for arg in vars(self.opt):
            self.extra_str += str(arg) + '_' + str(getattr(self.opt, arg))

        super(object, self).__init__()

    def test(self,seed):
        pre_classes_len=0
        last_task_name = str( self.nb_tasks)
        exp_dir = os.path.join(self.parent_exp_dir, last_task_name, 'CND/' + 'Seed' + str(seed) + self.extra_str)
        final_model_path = os.path.join(exp_dir, "best_model.pth.tar")
        for task in range(1, self.nb_tasks + 1):

            task_name = str(task)
            exp_dir = os.path.join(self.parent_exp_dir, task_name, 'CND/' + 'Seed' + str(seed) +  self.extra_str)
            model_path = os.path.join(exp_dir, "best_model.pth.tar")
            dataset_path = os.path.join(self.dataset_parent_dir, task_name, 'trainval_dataset.pth.tar')
            dsets = torch.load(dataset_path)

            if self.opt.shared_head:
                this_out_inx=-1
            else:
                this_out_inx = task - 1

            acc = test_model(final_model_path, dataset_path, shared_head=self.opt.shared_head,pre_classes_len=pre_classes_len,task_index=this_out_inx)
            accorg = test_model(model_path, dataset_path, shared_head=self.opt.shared_head,pre_classes_len=pre_classes_len, task_index=this_out_inx)
            print('task number ', task, ' accuracy is %.2f' % acc, 'compared to %.2f' % accorg)
            self.accs[task - 1] = acc
            self.forgettings[task - 1] = (accorg - acc)
            if self.opt.shared_head:
                pre_classes_len += len(dsets["test"].classes)

    def test_taskacc_per_model(self,seed):
        pre_classes_len=0
        last_task_name = str( self.nb_tasks)
        exp_dir = os.path.join(self.parent_exp_dir, last_task_name, 'CND/' + 'Seed' + str(seed) + self.extra_str)
        accs=[[0 for j in range(i+1) ] for i in range(self.nb_tasks + 1)]
        final_model_path = os.path.join(exp_dir, "best_model.pth.tar")
        for task_model in range(1, self.nb_tasks + 1):
            task_name = str(task_model)
            exp_dir = os.path.join(self.parent_exp_dir, task_name, 'CND/' + 'Seed' + str(seed) + self.extra_str)
            model_path = os.path.join(exp_dir, "best_model.pth.tar")
            pre_classes_len=0
            for task  in range(1, task_model+1):
                this_task_name = str(task)
                dataset_path = os.path.join(self.dataset_parent_dir, this_task_name, 'trainval_dataset.pth.tar')
                dsets = torch.load(dataset_path)

                if self.opt.shared_head:
                    this_out_inx=-1
                else:
                    this_out_inx = task - 1

                accorg = test_model(model_path, dataset_path, shared_head=self.opt.shared_head,pre_classes_len=pre_classes_len, task_index=this_out_inx)
                print('task number ', task_model, ' accuracy is %.2f' % accorg)
                accs[task_model - 1][task-1] = accorg
                if self.opt.shared_head:
                    pre_classes_len += len(dsets["test"].classes)
        return accs