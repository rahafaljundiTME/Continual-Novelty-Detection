import argparse
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from Tools.CN_Set import CN_Set
from Tools.Novelty_Moduls import CN_Model

"""
Construct the 3 In Out Forg. Sets for the TinyImageNet split.
"""
def construct_out(model_path, dataset_path, batch_size=100,task_index=0,pre_num_classes=0):
    """

    :param model_path:
    :param dataset_path:
    :param batch_size:
    :param task_index: in case of all heads in one model
    :return: a set indexes of correctly classified samples and a set of indesices of misclassified samples. Currently it is only based on the test set
    """
    if type(model_path) is str:
        model = torch.load(model_path)# the input is path
    else:
        model=model_path # the input is the model itself
    if isinstance(model, dict):
        model = model['model']
    model = model.cuda()
    model.eval()
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size,
                                                   shuffle=False, num_workers=4)
                    for x in ['train', 'val','test']}


    out_inds=np.array([]).astype(int)#only construct a set of misclassified examples here
    index=0
    if pre_num_classes>0:
        head_index=-1#shared head applies
    else:
        head_index=task_index
    for data in dset_loaders['test']:
        images, labels = data
        labels=labels+pre_num_classes
        images = images.cuda()
        images = images.squeeze()
        labels = labels.cuda()
        outputs = model(images)
        if isinstance(outputs,list):
            outputs=outputs[head_index]

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted != labels).squeeze().cpu().numpy()
        c=c.astype(bool)
        inds=np.array(list(range(index,index+images.size(0))))
        out_inds=np.append(out_inds,inds[c])

        index+=images.size(0)
        # pdb.set_trace()
    all_inds=list(range(index))
    in_inds=list(set(all_inds) -set(out_inds))
    return dsets["test"],in_inds,out_inds.tolist()


def construct_train_in(model_path, dataset_path, batch_size=100,task_index=0,pre_num_classes=0):
    """

    :param model_path:
    :param dataset_path:
    :param batch_size:
    :param task_index: in case of all heads in one model
    :return: a set indexes of correctly classified samples and a set of indesices of misclassified samples. Currently it is only based on the test set
    """
    if type(model_path) is str:
        model = torch.load(model_path)# the input is path
    else:
        model=model_path # the input is the model itself
    if isinstance(model, dict):
        model = model['model']
    model = model.cuda()
    model.eval()
    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size,
                                                   shuffle=False, num_workers=4)
                    for x in ['train', 'val','test']}


    out_inds=np.array([]).astype(int)#only construct a set of misclassified examples here
    index=0
    if pre_num_classes>0:
        head_index=-1#shared head applies
    else:
        head_index=task_index
    for data in dset_loaders['train']:
        images, labels = data
        labels=labels+pre_num_classes
        images = images.cuda()
        images = images.squeeze()
        labels = labels.cuda()
        outputs = model(images)
        if isinstance(outputs,list):
            outputs=outputs[head_index]

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted != labels).squeeze().cpu().numpy()
        c=c.astype(bool)
        inds=np.array(list(range(index,index+images.size(0))))
        out_inds=np.append(out_inds,inds[c])

        index+=images.size(0)
        # pdb.set_trace()
    all_inds=list(range(index))
    in_inds=list(set(all_inds) -set(out_inds))
    return dsets["train"],in_inds


parser = argparse.ArgumentParser()

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
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
n_seeds=10
seed=0# the model of the seed 0 is the one where computation should go
n_tasks=10#10
extra_str = ""#"tinyimagenet_exp"
for arg in vars(opt):
    extra_str = extra_str + str(arg, ) + '_' + str(getattr(opt, arg))

model_path=parent_exp_dir+'/'+str(n_tasks)+'/CND/'+'Seed'+str(seed) +extra_str+'/best_model.pth.tar'
data_dir='Data/TINYIMAGNET'
cn_models=[]#a lost of continual novelty detection models

for task_model in range(0,n_tasks):
    task_model_path = parent_exp_dir + str(task_model+1) + '/CND/' + 'Seed' + str(seed) + extra_str + '/best_model.pth.tar'
    current_model = torch.load(task_model_path)
    cnmodel = CN_Model(current_model)
    pre_num_classes = 0#this is not optimal as we are recomputing this over and over but it is fine
    for task in range(0,n_tasks):
        dataset_path = os.path.join(data_dir, str(task+1), 'trainval_dataset.pth.tar')

        if task < task_model:
            print("previous task")
            dataset, in_inds, out_inds = construct_out(current_model, dataset_path, task_index=task,pre_num_classes=pre_num_classes)
            #construct in set
            in_set = CN_Set(dataset, in_inds,pre_num_classes)
            cnmodel.In.append(in_set)
            #forgoteen samples should be constructed
            pre_cn_model=cn_models[task]
            forgotten_inds=list(set.intersection(set(pre_cn_model.In[task].indices),set(out_inds)))
            forgotten_set = CN_Set(dataset, forgotten_inds,pre_num_classes)
            cnmodel.Forgotten.append(forgotten_set)

        if task == task_model:
            print("current_task")
            #only inset applies
            dataset, in_inds, out_inds = construct_out(current_model, dataset_path, task_index=task,pre_num_classes=pre_num_classes)
            dataset_train, train_in_inds=construct_train_in(current_model, dataset_path, task_index=task,pre_num_classes=pre_num_classes)
            in_set = CN_Set(dataset, in_inds,pre_num_classes)
            train_in_set = CN_Set(dataset_train, train_in_inds, pre_num_classes)
            cnmodel.In.append(in_set)
            cnmodel.TrainIn=train_in_set
        if task > task_model:
            print("outset")
            #only outset applies
            dataset = torch.load(dataset_path)["test"]
            out_set = CN_Set(dataset, list(range(len(dataset))),pre_num_classes)
            cnmodel.Out.append(out_set)
        if opt.shared_head:
            dsets = torch.load(dataset_path)
            pre_num_classes+=len(dsets["test"].classes)
    cn_models.append(cnmodel)





torch.save(cn_models,"CN_MODELS"+extra_str+".pth")

