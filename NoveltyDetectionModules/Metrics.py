import torch
import numpy as np
"""
Calculate the different ND metrics.
"""

def in_as_out_err(out_prob,deltas):
    # calculate the amount of in from previous tasks considered as out when estiamting the best detection error of forgotten/in

    Y1 = out_prob

    error = 0
    for delta in deltas:

         error +=  np.sum(np.sum(Y1 < delta)) / np.float(len(Y1)) #in as out

    return error/len(deltas)

#In,out probs should be numpy array
def forg_err(in_prob, out_prob, num_classes,end):
    # calculate the minimium detection error of forgotten samples(out_prob) as out and that of pre-in(in_prob) as out
    start = 1 / num_classes
    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))
    gap = (end - start) / 100000
    Y1 =out_prob
    X1 = in_prob
    error = 1.0
    best_delta=start
    best_fnr=0
    best_fpr=0

    for delta in np.arange(start, end, gap):
        fnr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if error>((fnr + fpr) / 2.0):
            error =(fnr + fpr) / 2.0
            best_fnr=fnr#pre in as out
            best_fpr=fpr#forgoteen as in
            best_delta=delta

    return best_fnr,best_fpr,best_delta

def tpr95(in_prob, out_prob, num_classes,end):
    # calculate the falsepositive error when tpr is 95%
    # calculate baseline

    start=1/num_classes
    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))
    gap = (end - start) / 100000
    # if name == "CIFAR-10":
    #     end = 0.12, base=1
    # if name == "CIFAR-100":
    #     end = 0.0104,base=1

    Y1 = out_prob
    X1 = in_prob
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    if total==0:
        return 0
    fpr = fpr / total
    return fpr


def auroc(in_prob, out_prob, num_classes,end):
    #calculate the AUROC
    start = 1 / num_classes

    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))

    gap = (end- start)/100000
    Y1 = out_prob
    X1 = in_prob
    auroc = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        auroc += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    auroc += fpr * tpr

    return auroc


def auprIn(in_prob, out_prob, num_classes,end):
    # calculate the AUPR
    start = 1 / num_classes
    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))
    gap = (end - start) / 100000
    precisionVec = []
    recallVec = []
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = out_prob
    X1 =in_prob
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    return auprBase




def auprOut(in_prob, out_prob, num_classes,end):
    # calculate the AUPR

    start = 1 / num_classes
    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))
    gap = (end - start) / 100000
    Y1 =out_prob
    X1 = in_prob
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    return auprBase


def detection(in_prob, out_prob, num_classes,end):
    # calculate the minimum detection error

    start = 1 / num_classes
    end=max(max(in_prob),max(out_prob))#1/num_classes
    start=min(min(out_prob),min(in_prob))
    gap = (end - start) / 100000
    Y1 =out_prob
    X1 = in_prob
    error = 1.0
    best_deta=start
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        this_error = np.minimum(error, (tpr + error2) / 2.0)
        if this_error<error:
            best_deta=delta
            error=this_error
    return error,best_deta

