#

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from collections import namedtuple
from Continual_learning import Regularization_utils, Architecture

"""
Incremental Training using CL method (here could be MAS or EWC but others can be similarly integrated)
"""


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch > 0 and epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def set_lr(optimizer, lr, lr_decay, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count > 20:
        continue_training = False
        print("training terminated")
    # if count==10:
    #     lr = lr * lr_decay
    #     print('lr is set to {}'.format(lr))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    return optimizer, lr, continue_training


def traminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train_on_task(trainer, dataset_path, pre_dset, model_path, exp_dir, task_index=1, lr=0.0004,
                  lr_multiplier=None):
    num_epochs = trainer.opt.num_epochs
    batch_size = trainer.batch_size
    regularization_method = trainer.opt.regularization_method
    reg_lambda = trainer.opt.reg_lambda
    opt = trainer.opt
    lr_decay = trainer.lr_decay_rate
    print('lr is ' + str(lr))

    dsets = torch.load(dataset_path)
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                   shuffle=True, num_workers=0)
                    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['test'].classes
    for mode in ['train', 'val']:
        for data in dset_loaders[mode]:
            # cash the data first!
            print("next batch")

    dset_loaders[mode].dataset.dataset.set_use_cache(use_cache=True)
    dset_loaders[mode].num_workers = 4
    # import pdb;pdb.set_trace()
    # -----pre-dset loader
    if not pre_dset is None:
        # ---in case of a more suffisticated continual learning method, the loader can be overwritten
        pre_dset_loader = torch.utils.data.DataLoader(pre_dset, batch_size=batch_size,
                                                      shuffle=True, num_workers=0)
        pre_classes = sum(
            pre_dset.dataset.dataset_classes)  # subset with buffer indices and then it includes the dataset that has the info

        # cash the data first!
        for data in pre_dset_loader:
            print("next batch")
        for dataset in pre_dset_loader.dataset.dataset.datasets:  # ----getting through the layers to rich the original dataset
            dataset.dataset.set_use_cache(
                use_cache=True)  # -------crazy: so we have a loader on the concat dataset that has list of datasets and then each concate dataset is the training
            # subset of the Loaded Image Folder
        pre_dset_loader.num_workers = 4
    else:
        pre_dset_loader = None
        pre_classes = 0
    # -------------------

    if not os.path.isfile(model_path):
        num_classes = len(dset_classes)
        if opt.shared_head and opt.all_initialized:
            num_classes = num_classes * 10  # try initializing all tasks neurons first!

        if opt.arch == "ResNet":
            model = Architecture.ResNet18_dropout(num_classes, 40, (3, 56, 56),
                                                  dropr=opt.dropr)  # 20 is num of classes, num_features/8
            # model = Architecture.ResNet18_imprint(num_classes,nf= 40, input_size=(3, 56, 56))
        if opt.arch == "VGG":
            model = Architecture.VGGSlim2_dropout(num_classes=num_classes, dropr=opt.dropr)  # 20 is num of classes
        if opt.arch == "Alex":
            model = Architecture.alex_seq(pretrained=True, num_classes=num_classes,
                                          dropr=opt.dropr)  # 20 is num of classes

    else:
        model = torch.load(model_path)

    model, original_model = Regularization_utils.prepare_model(model, len(dset_classes), regularization_method,
                                                               task_index, opt)  # Multi head assumption applies now
    model.reg_lambda = reg_lambda

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    model = model.to(opt.device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if lr_multiplier is not None:
        # feature extraction run with lower learning rate
        lr_params = [{'params': model.module.features.parameters(), 'lr': lr_multiplier * lr},
                     {'params': model.module.classifier.parameters()}]
    else:
        lr_params = model.parameters()

    optimizer_ft = optim.SGD(lr_params, lr, momentum=0.9, weight_decay=0)
    # add the predatast here
    # import pdb;pdb.set_trace()
    model, best_acc = train_model_early_stopping(trainer, model, original_model, criterion, optimizer_ft, lr, lr_decay,
                                                 dset_loaders,
                                                 dset_sizes, pre_dset_loader, pre_classes, num_epochs, exp_dir,
                                                 regularization_method, task_index, opt)

    # model,best_acc = train_model(model, original_model,criterion, optimizer_ft, lr,lr_decay, dset_loaders,
    #                                              dset_sizes, pre_dset_loader,pre_classes,num_epochs, exp_dir,regularization_method,task_index,opt)
    model = Regularization_utils.model_post_process(model, dataset_path, regularization_method, opt=opt)
    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
    # remove check point
    # os.remove(os.path.join(exp_dir, 'epoch.pth.tar'))

    return model


def train_model_early_stopping(trainer, model, original_model, criterion, optimizer, lr, lr_decay, dset_loaders,
                               dset_sizes, pre_dset_loader, pre_classes, num_epochs, exp_dir='./',
                               regularization_method="LwF", task_index=0, opt=namedtuple("device", "cuda")):
    """
    Train a model with decaying learning rate and then early stopping based on the val set acc.
    """
    print('dictoinary length' + str(len(dset_loaders)))
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    # reg_params=model.reg_params
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    # train with exponential weight decay
    if opt.lr_scheduler == "plt":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    best_acc = 0.0

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        original_model = checkpoint['original_model']
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_acc = checkpoint['best_acc']
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 1
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))

    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, lr_decay, count=val_beat_counts)
                if not continue_training:
                    traminate_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)  # Set model to training mode

                if pre_dset_loader is not None:  # replay
                    predset_iter = iter(pre_dset_loader)


            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.squeeze()
                # correct the labels of the current task to match the shared head, in case. otherwise it will be zero
                labels = labels + pre_classes
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # ---get replayed sampeles if any
                if pre_dset_loader is not None:

                    try:
                        pre_batch = next(predset_iter)
                    except:
                        predset_iter = iter(pre_dset_loader)
                        pre_batch = next(predset_iter)
                    pre_inputs, pre_labels = pre_batch
                    pre_inputs = pre_inputs.squeeze()
                    pre_inputs, pre_labels = pre_inputs.to(opt.device), pre_labels.to(opt.device)
                    pre_batch = (pre_inputs, pre_labels)


                else:
                    pre_batch = None

                total_loss, task_loss, preds = Regularization_utils.generate_reg_loss(trainer, model, original_model,
                                                                                      outputs, labels, inputs,
                                                                                      criterion, pre_batch,
                                                                                      regularization_method, task_index,
                                                                                      pre_classes, opt)
                # print("total_loss",total_loss,"task_loss",task_loss)
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()
                # statistics
                running_loss += task_loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del task_loss
                    del preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
                    print("val_beat_counts is", str(val_beat_counts))
                if opt.lr_scheduler == "plt":
                    lr_scheduler.step(epoch_loss)
                else:
                    lr_scheduler.step()
        # epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'lr': lr,
            'val_beat_counts': val_beat_counts,
            'model': model,
            'original_model': original_model,
            'epoch_acc': epoch_acc,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc


# ===============exp lr schueler training
def train_model(model, original_model, criterion, optimizer, lr, lr_decay, dset_loaders, dset_sizes,
                pre_dset_loader, pre_classes, num_epochs, exp_dir='./', regularization_method="LwF",
                task_index=0, opt=namedtuple("device", "cuda")):
    """
    Train a model with exp schuedler, return model with best acc on val.
    """
    print('dictoinary length' + str(len(dset_loaders)))
    resume = os.path.join(exp_dir, 'epoch.pth.tar')
    # reg_params=model.reg_params
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not imporved
    # train with exponential weight decay
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        original_model = checkpoint['original_model']
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_acc = checkpoint['best_acc']
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 1
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))

    # pdb.set_trace()
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode

                if pre_dset_loader is not None:  # replay
                    predset_iter = iter(pre_dset_loader)


            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.squeeze()
                # correct the labels of the current task to match the shared head, in case. otherwise it will be zero
                labels = labels + pre_classes
                inputs, labels = inputs.to(opt.device), labels.to(opt.device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # ---get replayed sampeles if any
                if pre_dset_loader is not None:

                    try:
                        pre_batch = next(predset_iter)
                    except:
                        predset_iter = iter(pre_dset_loader)
                        pre_batch = next(predset_iter)
                    pre_inputs, pre_labels = pre_batch
                    pre_inputs = pre_inputs.squeeze()
                    pre_inputs, pre_labels = pre_inputs.to(opt.device), pre_labels.to(opt.device)
                    pre_batch = (pre_inputs, pre_labels)


                else:
                    pre_batch = None

                total_loss, task_loss, preds = Regularization_utils.generate_reg_loss(model, original_model, outputs,
                                                                                      labels, inputs, criterion,
                                                                                      pre_batch, regularization_method,
                                                                                      task_index, opt)
                print("total_loss", total_loss, "task_loss", task_loss)
                if phase == 'train':
                    total_loss.backward()
                    optimizer.step()
                # statistics
                running_loss += task_loss.data.item()
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del task_loss
                    del preds
                    best_acc = epoch_acc
                    torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1
                    print("val_beat_counts is", str(val_beat_counts))
                lr_scheduler.step()
        # epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'lr': lr,
            'val_beat_counts': val_beat_counts,
            'model': model,
            'original_model': original_model,
            'epoch_acc': epoch_acc,
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model, best_acc


# ===================end of exp lr schuedler training
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # best_model = copy.deepcopy(model)
    torch.save(state, filename)
# len(dset_classes)
