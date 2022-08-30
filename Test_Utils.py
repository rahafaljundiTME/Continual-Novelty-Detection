"""
Util functions for test, plots, etc..
"""
#Main test functions
from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
import pylab


def test_model(model_path, dataset_path, shared_head,pre_classes_len,batch_size=100, print_classes_acc=False,task_index=0):

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
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val','test']}

    dset_classes = dsets['test'].classes
    class_correct = list(0. for i in range(len(dset_classes)))
    class_total = list(0. for i in range(len(dset_classes)))
    for data in dset_loaders['test']:
        images, labels = data
        images = images.cuda()
        images = images.squeeze()
        if shared_head:
            labels=labels+pre_classes_len
        labels = labels.cuda()
        outputs = model(images)
        if isinstance(outputs,list):

            outputs=outputs[task_index]# if shared head this will be -1
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        # pdb.set_trace()
        for i in range(len(predicted)):
            label = labels[i]
            class_correct[label-pre_classes_len] += c[i].item()
            class_total[label-pre_classes_len] += 1
        del images
        del labels
        del outputs
        del data
    if print_classes_acc:

        for i in range(len(dset_classes)):
            print('Accuracy of %5s : %2d %%' % (
                dset_classes[i], 100 * class_correct[i] / class_total[i]))
    accuracy = np.sum(class_correct) * 100 / np.sum(class_total)
    print('Accuracy: ' + str(accuracy))
    return accuracy

# def get_task_model(previous_model_path,current_model_path):
#     """
#
#     :param previous_model_path:
#     :param current_model_path:
#     :return: the corresponding task model
#     """
#     current_model_ft=torch.load(current_model_path)
#     if isinstance(current_model_ft, dict):
#         current_model_ft=current_model_ft['model']
#     if isinstance(current_model_ft, Lwf_model):
#         print("LWF MODEL TEST")
#
#     else:
#         previous_model_ft=torch.load(previous_model_path)
#         if isinstance(previous_model_ft, dict):
#             previous_model_ft=previous_model_ft['model']
#
#         last_layer_index=str(len(previous_model_ft.classifier._modules)-1)
#
#         current_model_ft.classifier._modules[last_layer_index] = previous_model_ft.classifier._modules[last_layer_index]
#
#     return current_model_ft
#
# def test_seq_task_performance(previous_model_path,current_model_path,dataset_path,task_index=0):
#
#     current_model_ft=get_task_model(previous_model_path,current_model_path)
#
#     acc=test_model(current_model_ft,dataset_path,task_index=task_index)
#     return acc

def plot_scatter_plot(x_values, colors, labels, markers,save_img=False,legend="out",ylabel="Accuracy"):
    # create plot
    fig, ax = plt.subplots()
    plt.grid(True, alpha=0.3)
    for kind in x_values.keys():

        pylab.plot(list(range(len(x_values[kind]))),x_values[kind], colors[kind], label=kind,marker=markers[kind], linewidth=2)
    pylab.legend(loc='best', prop={'size': 13})
    if save_img:
        pylab.axis('on')
        pylab.tight_layout()
        # plt.gca().set_position([0, 0, 1, 1])
        # pylab.savefig('{}.svg'.format(save_img) , bbox_inches='tight')
        pylab.savefig('{}.png'.format(save_img), bbox_inches='tight')
        pylab.clf()
    else:
        pylab.show()

    return
def plot_multibar(seqacc, keys, colors, labels, hatches, save_img=False, ylim=(88, 97), bar_widthT=0.1, bar_width=0.1,
                  legend="out",ylabel="Accuracy"):
    # data to plot
    fig_size = plt.rcParams["figure.figsize"] 
    fig_size[0] = 14
    fig_size[1] = 5   
    plt.rcParams["figure.figsize"] = fig_size
    terminate=True
    for key in keys:
        if len(seqacc[key])>0:
            terminate=False
            break
    if terminate:
        return
    n_groups = len(keys)

    # create plot
    fig, ax = plt.subplots()
    index = np.array([0, 1.3, 2.5, 3.6, 4.7])

    index2 = np.array([0, 1.3, 2.5, 3.6, 4.7, 5.9, 7.1, 8.2, 9.5])

    opacity = 0.8

    plt.grid(True, alpha=0.3)
    xx = 0
    methindex = 0
    #fig_size = plt.rcParams["figure.figsize"]
    #fig_size[0] = 14
    #fig_size[1] = 5
    #plt.rcParams["figure.figsize"] = fig_size
    X = np.arange(len(seqacc[keys[0]]))
    xticks = ['S' + str(x + 1) for x in X]
    
    for key in keys:
        
        pylab.bar(X + methindex * bar_width, seqacc[key], width=bar_widthT, color=colors[methindex],
                  label=labels[methindex], hatch=hatches[methindex], alpha=0.9)
        for index, value in enumerate(seqacc[key]):
            mult=1 if value>1 else 100
            #removed for plutting norm,features,std
            #pylab.text(index + methindex * bar_width -bar_width/2,value, str(((round(value*mult,1)))),fontsize=9)
        methindex += 1

    pylab.xticks(X + (methindex / 3) * bar_width, xticks, fontsize=20, color='black')
    # pylab.legend(loc='best', prop={'size': 13})
    if not legend is None:
        if legend == "out":
            pylab.legend(bbox_to_anchor=(1.0, 1), loc="upper left", prop={'size': 15})
        else:
            #ylim[1]-.08 ,,frameon=False for Alex
            pylab.legend(bbox_to_anchor=(0.5, 1),loc='upper center',ncol=5, prop={'size': 12})#18
            #plt.legend(frameon=False)
    pylab.ylabel(ylabel, fontsize=14)

    pylab.ylim(ylim)
    # pylab.legend(loc='upper left', bbox_to_anchor=(1, 1))

    pylab.tight_layout()
    pylab.tick_params(axis='both', which='major', labelsize=20)
    if save_img:
        pylab.axis('on')
        pylab.tight_layout()
        # plt.gca().set_position([0, 0, 1, 1])
        # pylab.savefig('{}.svg'.format(save_img) , bbox_inches='tight')
        pylab.savefig('{}.png'.format(save_img), bbox_inches='tight')
        pylab.savefig('{}.pdf'.format(save_img), bbox_inches='tight')
        pylab.clf()
    else:
        pylab.show()

class Options(object):
    #wrapper to pass options
    def __init__(self):
        super(Options, self).__init__()

def get_methods_best_results(novelty_methods,opts):

    import glob
    best_results = {}
    best_results_name = {}
    for novelty_method in novelty_methods:
        # "Results" + novelty_extra_str + "_" + str(opt.arch) + "_" + str(opt.regularization_method) + "_" + str(
        #     opt.reg_lambda) + ".pth"

        main_result_file =opts.main_result_file #"Resultstinyimagenet_exp"

        main_result_file += "novelty_method_" + str(novelty_method) + "*" + "novelty_tuned_on_in_task_False_" + str(
            opts.buffer_size) + "_" + str(opts.arch) + \
                            "_" + str(opts.method) + "_" + str(opts.reg_lambda) + ".pth"
        filename = "" + main_result_file  # + "*.pth"
        # print(filename)
        files_path = glob.glob(filename, recursive=True)
        # print(files_path)
        min_out_pr = 1
        max_out_auc = 0
        min_results_file = ""
        this_best_results = None
        for file in files_path:
            results = torch.load(file)

            out_pr = 0
            # for result in results:# this can be reverted to only the first task
            # only on the first task
            result = results[0]
            if opts.tune_baed_on == "out":
                # if result.out_data_tpr==0:
                #
                #     out_pr = 1
                #     break

                out_pr += result.out_data_tpr  # /len(results)
                out_auc = result.out_data_auc
            else:
                # if len(result.det_errs)>0 and len(result.det_errs[0])>0:
                out_pr += result.det_errs[0][0]  # /len((results)-1)
            # if min_out_pr > out_pr and out_pr!=0:
            if max_out_auc < out_auc:
                # import pdb;pdb.set_trace()
                min_results_file = file
                this_best_results = results
                # min_out_pr = out_pr
                max_out_auc = out_auc
                print(novelty_method, out_auc)

        if this_best_results is not None:
            best_results[novelty_method] = this_best_results
            best_results_name[novelty_method] = min_results_file
            print("best results novelty method", novelty_method, min_results_file)


    return best_results
