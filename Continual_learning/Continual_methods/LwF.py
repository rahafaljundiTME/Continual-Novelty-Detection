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


def Fdistillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    #loss = F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1)) * scale

    lossH = -torch.mean(torch.sum(F.log_softmax(y / T, dim=1)*F.softmax(teacher_scores / T, dim=1),dim=1))

    return lossH



def Rdistillation_loss(y, teacher_scores, T, scale):
    p_y = F.softmax(y)
    p_y = p_y.pow(1 / T)
    sumpy = p_y.sum(1)
    sumpy = sumpy.view(sumpy.size(0), 1)
    p_y = p_y.div(sumpy.repeat(1, scale))
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    loss = -p_teacher_scores * torch.log(p_y)
    loss = loss.sum(1)

    loss = loss.sum(0) / loss.size(0)
    return loss


def distillation_loss(y, teacher_scores, T, scale):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """

    maxy, xx = y.max(1)
    maxy = maxy.view(y.size(0), 1)
    norm_y = y - maxy.repeat(1, scale)
    ysafe = norm_y / T
    exsafe = torch.exp(ysafe)
    sumex = exsafe.sum(1)
    ######Tscores
    maxT, xx = teacher_scores.max(1)
    maxT = maxT.view(maxT.size(0), 1)
    teacher_scores = teacher_scores - maxT.repeat(1, scale)
    p_teacher_scores = F.softmax(teacher_scores)
    p_teacher_scores = p_teacher_scores.pow(1 / T)
    p_t_sum = p_teacher_scores.sum(1)
    p_t_sum = p_t_sum.view(p_t_sum.size(0), 1)
    p_teacher_scores = p_teacher_scores.div(p_t_sum.repeat(1, scale))
    #  Y = sum(sum(sum(log(sumex) - sum(c .* x_safe,3),1),2),4) ;

    loss = torch.sum(torch.log(sumex) - torch.sum(p_teacher_scores * ysafe, 1))

    loss = loss / teacher_scores.size(0)
    return loss




