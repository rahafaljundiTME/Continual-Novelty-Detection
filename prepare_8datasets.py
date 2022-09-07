import random

from torchvision import transforms
#from data.imgfolder import *

import bisect
import os
import os.path

from PIL import Image
import numpy as np
import copy
from itertools import accumulate

import torch
import torch.utils.data as data
from torchvision import datasets
from Tools.CN_Set import ImageFolderTrainVal
# SCENES
# Data augmentation and normalization for training
# Just normalization for validation
# I have to make sure if for alexnet it is 224 or 227 here
# import Lambda
data_transforms = {

    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dirs={}
annotation_dirs={}
#test_lists={}
datasets_path="./8seq/datasets" #PLEASE DOWNLOD THE 8 SEQUENCE DATASETS AND ADJUST datasets_path
data_dirs["1"]=datasets_path+"/Flower/Images"
data_dirs["2"]=datasets_path+"/Scenes/Images"
data_dirs["3"]=datasets_path+"/CUB_200_2011/images"
data_dirs["4"]=datasets_path+"Pytorch_Cars_dataset/images"
data_dirs["5"]=datasets_path+"Pytorch_AirCraft_dataset/images"
data_dirs["6"]=datasets_path+"Pytorch_Actions_dataset/images"
data_dirs["7"]=datasets_path+"Pytorch_Letters_dataset/images"
data_dirs["8"]=datasets_path+"Pytorch_SVHN_dataset/images"
annotation_dirs["1"]= datasets_path+"datasets/Flower"
annotation_dirs["2"]= datasets_path+"datasets/scenes"
annotation_dirs["3"]= datasets_path+"datasets/CUB11f"
annotation_dirs["4"]= datasets_path+"datasets/Cars"
annotation_dirs["5"]= datasets_path+"datasets/AirCraft"
annotation_dirs["6"]= datasets_path+"datasets/Actions"
annotation_dirs["7"]= datasets_path+"datasets/Letters"
annotation_dirs["8"]= datasets_path+"datasets/SVHN"
exp_data_dir="./8seq"
for task in list(data_dirs.keys()):
    data_dir =data_dirs[task]
    train_list = os.path.join(annotation_dirs[task],"TrainImages.txt")
    test_list = os.path.join(annotation_dirs[task],"TestImages.txt")
    all_files_list = [train_list, test_list]
    modes = ['train', 'val']
    initial_dsets = {modes[x]: ImageFolderTrainVal(data_dir, all_files_list[x], data_transforms[modes[x]])
             for x in [0, 1]}
    train_dataset, val_dataset = torch.utils.data.random_split(initial_dsets["train"], [int(len(initial_dsets["train"])*0.8) , len(initial_dsets["train"])-int(len(initial_dsets["train"])*0.8)])
    dsets={}
    dsets['train']=train_dataset

    dsets['val'] = val_dataset
    dsets['test'] = initial_dsets["val"]
    output_dir=os.path.join(exp_data_dir,task)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    torch.save(dsets,os.path.join(output_dir,'trainval_dataset.pth.tar'))

