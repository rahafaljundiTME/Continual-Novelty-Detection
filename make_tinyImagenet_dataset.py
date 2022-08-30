import torch
from torchvision import datasets, transforms
from Tools.CN_Set  import LoadedImageFolderDataset


"""
create the tinyimagetnet split sequence
"""


import os
n_tasks=10
# Data loading code
data_dir='./Data/TINYIMAGNET/'
for task in range(1,n_tasks+1):
    traindir = os.path.join(data_dir, str(task),'train')
    testdir = os.path.join(data_dir, str(task), 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    trainval_dataset = LoadedImageFolderDataset(
            traindir,
            transforms.Compose([
                transforms.Resize(64),
                transforms.RandomResizedCrop(56),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
    ]))

    test_dataset=LoadedImageFolderDataset(testdir, transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                normalize,
    ]))
    #the train and val datasets will be augmented.
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [int(len(trainval_dataset)*0.8) , int(len(trainval_dataset)*0.2)])
    dsets={}
    dsets['train']=train_dataset
    dsets['val']=val_dataset
    dsets['test'] = test_dataset

    torch.save(dsets,os.path.join(data_dir,str(task),'trainval_dataset.pth.tar'))    




