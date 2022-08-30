from __future__ import print_function, division

import torch
import torch.nn.functional as F


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


criterion = torch.nn.CrossEntropyLoss(reduction='sum')


def loss(original_model, task_index, pre_input, tasks_outputs, labels, pre_outputs, pre_label, pre_classes,
         seen_classes, T=2):
    extended_seen_classes = [0] + seen_classes
    tasknum = task_index - 1

    # try to get the below information

    end = tasks_outputs[0].size(1)  # check!
    mid = pre_classes  # self.seen_classes[-1]
    batch_size = tasks_outputs[0].size(0)
    replay_size = pre_outputs.size(0)
    # ============================
    start = 0

    # curr, prev = samples
    #
    # data, target = curr
    #
    # target = target % (end - mid)
    #
    # batch_size = data.shape[0]
    # data_r, target_r = prev
    # replay_size = data_r.shape[0]
    # data, data_r = data.cuda(), data_r.cuda()
    # data = torch.cat((data, data_r))
    # target, target_r = target.cuda(), target_r.cuda()

    # output = self.model(data)
    loss_KD = 0

    loss_CE_curr = 0
    loss_CE_prev = 0

    curr = tasks_outputs[0][:, mid:end]
    shifted_labels = labels - mid
    loss_CE_curr = criterion(curr, shifted_labels)

    prev = pre_outputs[:, start:mid]
    loss_CE_prev = criterion(prev, pre_label)
    loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

    # loss_KD
    score = original_model(pre_input)[0][:, :mid].data  # ----------------I have to find this!:(((((((((((((((((
    loss_KD = torch.zeros(tasknum).cuda()
    start_KD = 0
    print("seen_classes", extended_seen_classes)
    for t in range(tasknum):
        start_KD += extended_seen_classes[t]
        end_KD = start_KD + extended_seen_classes[t + 1]
        print("task", t)
        soft_target = F.softmax(score[:, start_KD:end_KD] / T, dim=1)
        output_log = F.log_softmax(pre_outputs[:, start_KD:end_KD] / T, dim=1)
        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)
    loss_KD = loss_KD.sum()

    total_loss = loss_KD + loss_CE

    return total_loss, loss_CE_curr