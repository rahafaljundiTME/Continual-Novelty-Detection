

import argparse

from Continual_learning.Architecture import *
from Continual_learning.continual_learning_traintest import Continual_Tester
from Test_Utils import *
import traceback
import os
import numpy as np
"""
test the continual learning performance on TinyImageNet sequence
"""
parser = argparse.ArgumentParser()

#parser.add_argument('--num_epochs', type=int, default=100, help='training number of epochs')
#parser.add_argument('--reg_lambda', type=float, default=4)
#parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
#parser.add_argument('--b1', type=bool, default=False, help='online')
#parser.add_argument('--arch',  choices=['ResNet', 'VGG'], default='ResNet', help='backbone architicture')
#parser.add_argument('--dropout', type=bool, default=False, help='dropout enabled or disabled?')
#parser.add_argument('--device', type=str, default='cuda:0', help='which gpu, default is cuda:0?')
#parser.add_argument('--no_bias', action='store_true', help='if it is multi head the bias will be turned off') 
#parser.add_argument('--regularization_method', choices=['MAS', 'LwF'], default='MAS', help='which regularization  method should be used for continual learning?')
#parser.add_argument('--shared_head',  action='store_true', help='is it a shared head experminet with one shared output layer among all tasks?')
#parser.add_argument('--all_initialized',  action='store_true', help='if it is a shared head, this defines the initialization of the neurons')
#parser.add_argument('--buffer_size', type=int, default=0, help='buffer size for replayed samples')
#parser.add_argument('--batch_size', type=int, default=64, help='buffer size for replayed samples')
#parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='buffer size for replayed samples')
#parser.add_argument('--dropr', type=float, default=0.5, help='dropout rate')
#parser.add_argument('--lr_scheduler',  choices=['exp', 'plt'], default='plt', help='learning rate scheduler')
parser.add_argument('--num_epochs', type=int, default=100, help='training number of epochs')
parser.add_argument('--reg_lambda', type=float, default=4)
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--b1', type=bool, default=False, help='online')
parser.add_argument('--arch',  choices=['ResNet', 'VGG'], default='ResNet', help='backbone architicture')
#parser.add_argument('--dropout', type=bool, default=False, help='dropout enabled or disabled?')
parser.add_argument('--device', type=str, default='cuda:0', help='which gpu, default is cuda:0?')
parser.add_argument('--regularization_method', choices=['MAS', 'LwF','SSIL'], default='MAS', help='which regularization  method should be used for continual learning?')
parser.add_argument('--shared_head',  action='store_true', help='is it a shared head experminet with one shared output layer among all tasks?')
parser.add_argument('--all_initialized',  action='store_true', help='if it is a shared head, this defines the initialization of the neurons')
parser.add_argument('--no_bias',  action='store_true', help='if it is multi head the bias will be turned off')
parser.add_argument('--buffer_size', type=int, default=0, help='buffer size for replayed samples')
parser.add_argument('--batch_size', type=int, default=64, help='buffer size for replayed samples')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='buffer size for replayed samples')
parser.add_argument('--dropr', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr_scheduler',  choices=['exp', 'plt'], default='plt', help='learning rate scheduler')



opt = parser.parse_args()



###############################################

results={}#torch.load("tinyImagenet.pth")

parent_exp_dir='./TINYIMAGNET_exp_dir/'#Change to yours



print("***** reg_lambda is ",opt.reg_lambda,"**********")
n_seeds=1
n_tasks=10
accs=[[0 for i in range(n_seeds)] for j in range(n_tasks)]
forgettings=[[0 for i in range(n_seeds)] for j in range(n_tasks)]
#extra_str="tinyimagenet_exp"+"num_epochs_60reg_lambda_"+str(reg_lambda)+"lr_0.01b1_Falseneuron_omega_Truenormalize_"+str(normalize)+"dropout_Falsescale_"+str(scale)+"_lam"+str(lam)
dataset_parent_dir = 'Data/TINYIMAGNET'
continual_tester = Continual_Tester(n_tasks, opt, parent_exp_dir, dataset_parent_dir)
models_accs=continual_tester.test_taskacc_per_model(0)

torch.save(models_accs,"permodel_task_acc_"+continual_tester.extra_str+".pth")
if True:#temporary change to TRUE
    for seed in range(n_seeds):

        #models will be saved on  the exp dir
        continual_tester.test(seed)
        for task in range(n_tasks):
            accs[task][seed] = continual_tester.accs[task]
            forgettings[task ][seed] = continual_tester.forgettings[task]
    mean_acc=[np.mean(acc_task) for acc_task in accs]
    std_acc=[np.std(acc_task) for acc_task in accs]
    mean_forgetting=[np.mean(forg_task) for forg_task in forgettings]
    std_forgetting=[np.std(forg_task) for forg_task in forgettings]
    results["accs"]=accs
    results["forgettings"]=forgettings
    results["mean_acc"] = mean_acc
    results["std_acc"]=std_acc
    results["mean_forgetting"]=mean_forgetting
    results["std_forgetting"]=std_forgetting


torch.save(results,continual_tester.extra_str+".pth")


#
# seqacc={}
# keys=list(results.keys())
# avg={}
# for key in keys:
#     seqacc[key]=results[key][0]
#     avg[key]=sum(seqacc[key])/len(seqacc[key])
#
# hatches=["",""]
# colors=['C0','C2']
# labels=['SNID: '+str(round(avg[keys[0]],2)),"No-Reg: "+str(round(avg[keys[1]],2))]
#
# plot_multibar( seqacc,keys,colors,labels,hatches,save_img="tinyimagent_bars",ylim=(40, 66),bar_widthT= 0.12,bar_width = 0.14,legend="best")
#
# torch.save(results,"SampleCode_tinyImagenet_results.pth")
res=[]
ores=[]
[print( "model", i, "task", j,"naive",sum(res[i].det_errs[j]) / len(res[i].det_errs[j]),"norm_deltaMgp1",sum(ores[i].det_errs[j]) / len(res[i].det_errs[j])) for i in range(len(res)) for j in range(len(res[i].det_errs.keys()))]
