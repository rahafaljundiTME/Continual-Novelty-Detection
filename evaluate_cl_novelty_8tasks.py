import argparse
import os
import torch
import os.path
import numpy as np
from torchvision import datasets, transforms
from novelty_detection_evaluator import CNV_Evaluator,Tuned_CNV_Evaluator
"""
Evaluate CND performance on 8-task sequence
"""
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
parser.add_argument('--dropr', type=float, default=0.0, help='dropout rate')
parser.add_argument('--lr_scheduler',  choices=['exp', 'plt'], default='plt', help='learning rate scheduler')

#-------------------here goes novelty detection parameters------------------------------------
parser.add_argument('--novelty_method', choices=['ODIN', 'Max_Softmax','Mahalanobis','Gen_based','Energy','Open_Sim','Open_Sim_perturb'], default='ODIN', help='which novelty detection to use')
parser.add_argument('--novelty_magnitude', default=0.0014, type=float,
                    help='ODIN perturbation magnitude')
parser.add_argument('--novelty_temperature', default=1000, type=int,
                    help='ODIN temperature scaling')
parser.add_argument('--novelty_batchsize', type=int, default=200, help='batchsize for training novelty detection module')
parser.add_argument('--novelty_warmup', default=1, type=int,
                    help='VAE warmup')
parser.add_argument('--novelty_feature_based', action='store_true')

parser.add_argument('--novelty_z_size', default=32, type=int,
                    help='VAE z_size')

parser.add_argument('--novelty_layer_index', default=0, type=int,
                    help='layer index for feature extraction for the novelty method')
parser.add_argument('--novelty_max_beta', default=1, type=int,
                    help='VAE Beta')
parser.add_argument('--novelty_n_epochs', default=100, type=int,
                    help='number of epochs for training the generated model')
parser.add_argument('--novelty_tuned_on_in_task', action='store_true')

opt = parser.parse_args()



###############################################

results={}#torch.load("tinyImagenet.pth")

parent_exp_dir='./8seq_exp_dir/'#Change to yours



print("***** reg_lambda is ",opt.reg_lambda,"**********")

n_seeds=10
seed=0# the model of the seed 0 is the one where computation should go
n_tasks=8#10
extra_str = ""#"tinyimagenet_exp"
for arg in vars(opt):
    if not 'novelty' in arg:
        extra_str = extra_str + str(arg, ) + '_' + str(getattr(opt, arg))
novelty_extra_str="8tasks_exp"
for arg in vars(opt):
    if  'novelty' in arg:
        if(not "novelty_warmup"  in arg) and (not "novelty_layer_index" in arg):

            novelty_extra_str = novelty_extra_str + str(arg, ) + '_' + str(getattr(opt, arg))
opt.data_mean=[0.485, 0.456, 0.406]
opt.data_std=[0.229, 0.224, 0.225]
data_dir='./8seq/'
cn_models=[]#a lost of continual novelty detection models
cn_models_path = "cnmodels/CN_MODELS"+extra_str+".pth"
cn_models=torch.load(cn_models_path, map_location= 'cpu')
data_sequence=[]
for task in range(0,n_tasks):
    dataset_path = os.path.join(data_dir, str(task+1), 'trainval_dataset.pth.tar')
    data_sequence.append(dataset_path)

output_name="Results"+novelty_extra_str+"_"+str(opt.buffer_size)+"_"+str(opt.arch)+"_"+str(opt.regularization_method)+"_"+str(opt.reg_lambda)+".pth"

res = []
nvmodel = None
if os.path.isfile(output_name):
    res = torch.load(output_name)
    nvmodel = torch.load("NVMODEL_" + output_name)

if opt.novelty_tuned_on_in_task:
    output_name = "TUNED" + output_name

    cnv_evaluate = Tuned_CNV_Evaluator(cn_models, data_sequence, opt)

else:

    cnv_evaluate = CNV_Evaluator(cn_models, data_sequence, opt)  #

cnv_evaluate.output_name = output_name
all_task_results = cnv_evaluate.evaluate(opt, res, nvmodel)

torch.save(all_task_results,output_name)