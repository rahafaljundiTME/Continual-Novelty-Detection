import argparse
import os
import torch
from Continual_learning import Regularized_Training
from Continual_learning.continual_learning_traintest import Continual_Trainer
from Tools.CN_Set import  ImageFolderTrainVal
def set_random(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)


import random
import numpy

# GETTING INPUT PARAMS
# --------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='training number of epochs')
parser.add_argument('--reg_lambda', type=float, default=4)
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--b1', type=bool, default=False, help='online')
parser.add_argument('--arch',  choices=['ResNet', 'VGG',"Alex"], default='Alex', help='backbone architicture')
#parser.add_argument('--dropout', type=bool, default=False, help='dropout enabled or disabled?')
parser.add_argument('--device', type=str, default='cuda:0', help='which gpu, default is cuda:0?')
parser.add_argument('--regularization_method', choices=['MAS', 'LwF'], default='MAS', help='which regularization  method should be used for continual learning?')
parser.add_argument('--shared_head',  action='store_true', help='is it a shared head experminet with one shared output layer among all tasks?')
parser.add_argument('--all_initialized',  action='store_true', help='if it is a shared head, this defines the initialization of the neurons')
parser.add_argument('--no_bias',  action='store_true', help='if it is multi head the bias will be turned off')
parser.add_argument('--buffer_size', type=int, default=0, help='buffer size for replayed samples')
parser.add_argument('--batch_size', type=int, default=200, help='buffer size for replayed samples')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='buffer size for replayed samples')
parser.add_argument('--dropr', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr_scheduler',  choices=['exp', 'plt'], default='plt', help='learning rate scheduler')

opt = parser.parse_args()
num_epochs = opt.num_epochs
reg_lambda = opt.reg_lambda
lr = opt.lr
arch = opt.arch
b1 = opt.b1
#dropout = opt.dropout
lr_decay_epoch = 20

# --------------------------------------
#--------------------------------------

for seed in range(5):
    set_random(seed=seed)
    nb_tasks = 8
    model_path =''#''./vgg11slim2.pth.tar'
    parent_exp_dir = './8seq_exp_dir/'
    dataset_parent_dir = '/home/ral3833/8seq/'
    continual_trainer=Continual_Trainer(nb_tasks ,opt,parent_exp_dir,dataset_parent_dir,seed)
    #models will be saved on  the exp dir
    continual_trainer.train()
    # for task in range(1, nb_tasks + 1):
    #     task_name = str(task)
    #     extra_str = "tinyimagenet_exp"
    #     for arg in vars(opt):
    #         extra_str = extra_str + str(arg, ) + '_' + str(getattr(opt, arg))
    #
    #     print(extra_str)
    #
    #
    #     parent_exp_dir = './TINYIMAGNET_exp_dir/'
    #     d
    #
    #     exp_dir = os.path.join(parent_exp_dir, task_name, 'CND/'+'Seed'+str(seed) + extra_str)
    #     dataset_path = os.path.join(dataset_parent_dir, task_name, 'trainval_dataset.pth.tar')
    #     set_random(seed=seed)
    #     Regularized_Training.train_on_task(dataset_path=dataset_path, num_epochs=num_epochs, exp_dir=exp_dir,task_index=task, model_path=model_path,
    #                                        batch_size=200, lr=lr, weight_decay=0,regularization_method=opt.regularization_method, reg_lambda=reg_lambda,opt=opt)
    #     model_path = os.path.join(exp_dir,"best_model.pth.tar")
