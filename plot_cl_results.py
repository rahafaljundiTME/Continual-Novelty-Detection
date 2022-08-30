import torch
from Test_Utils import *

barwidthT = 0.18
barwidth = 0.20
if 1:
    Model1 = torch.load( "continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.0001b1_Falsearch_Alexdevice_cuda:0regularization_method_LwFshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_200lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")
    #Model2 = torch.load(
    #    "permodel_task_acc_num_epochs_100reg_lambda_20.0lr_0.0001b1_Falsearch_Alexdevice_cuda:0regularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_200lr_decay_rate_0.1dropr_0.5lr_scheduler_plt.pth")
    Model2=torch.load("continuallearning_res/num_epochs_100reg_lambda_20.0lr_0.0001b1_Truearch_Alexdevice_cuda:0regularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_200lr_decay_rate_0.1dropr_0.5lr_scheduler_plt.pth")
    Model3 = torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.0001b1_Falsearch_Alexdevice_cuda:0regularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_200lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")

    results = {}
    results["LwF"] = Model1["mean_acc"]
    results["MAS"] = Model2["mean_acc"]
    results["Finetune"] = Model3["mean_acc"]

    hatches = ["", "", ""]
    colors = ['C0', 'C2', 'C4']
    labels =list(results.keys()) #['ER Buffer 9k', 'ER Buffer 4.5k', 'ER Buffer 1.8k']
    keys = list(results.keys())
    plot_multibar(results, keys, colors, labels, hatches,
                  save_img="plots/cl_results/8tasks_multhead" , ylim=(0, 90),bar_widthT=barwidthT, bar_width=barwidth,legend="best")
        # FINE_RESNET_permodel = torch.load(
        #     "num_epochs_100reg_lambda_0.0lr_0.01b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Trueall_initialized_Falsebuffer_size_5000batch_size_128lr_decay_rate_0.9dropr_0.2lr_scheduler_exp.pth")
        #
        # for task_model in range(len(FINE_RESNET_permodel)):
        #     this_model_acc=FINE_RESNET_permodel[task_model]
        #     plot_multibar(this_model_acc, keys, colors, labels, hatches,
        #                   save_img="23_10_plots/TINYIMAGENET_"+"TASKMODEL_"+str(task_model)+"_" + arch + "_MAS_LaMBDA" + str(mas_reg_lambda),
        #                   ylim=(0, 68), bar_widthT=0.12, bar_width=0.14, legend="best")

if 1:#this is for shared head
    for arch in ['ResNet']:
        for b1 in ['False']:
            Model1 = torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.05b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Trueall_initialized_Falsebuffer_size_9000batch_size_16lr_decay_rate_0.9dropr_0.2lr_scheduler_exp.pth")
            Model2 = torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.05b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Trueall_initialized_Falsebuffer_size_1800batch_size_16lr_decay_rate_0.9dropr_0.2lr_scheduler_exp.pth")
            Model3=torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.05b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Trueall_initialized_Falsebuffer_size_4500batch_size_16lr_decay_rate_0.9dropr_0.2lr_scheduler_exp.pth")
            for mas_reg_lambda in [0.0]:
                results={}
                results["ER_buff9k"] = Model1["mean_acc"]
                results["ER_buff4.5k"]= Model3["mean_acc"]
                results["ER_buff1.8k"] = Model2["mean_acc"]

                hatches=["","",""]
                colors=['C0','C2','C4']
                labels=['ER Buffer 9k','ER Buffer 4.5k','ER Buffer 1.8k']
                keys=list(results.keys())
                plot_multibar(results,keys, colors, labels, hatches, save_img="plots/cl_results/sharedhead_plts_TINYIMAGENET_"+arch+"_MAS_LaMBDA"+str(mas_reg_lambda)+'B1'+b1,ylim=(0, 80),bar_widthT=barwidthT, bar_width=barwidth,legend="best")
            # FINE_RESNET_permodel = torch.load(
            #     "num_epochs_100reg_lambda_0.0lr_0.01b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Trueall_initialized_Falsebuffer_size_5000batch_size_128lr_decay_rate_0.9dropr_0.2lr_scheduler_exp.pth")
            #
            # for task_model in range(len(FINE_RESNET_permodel)):
            #     this_model_acc=FINE_RESNET_permodel[task_model]
            #     plot_multibar(this_model_acc, keys, colors, labels, hatches,
            #                   save_img="23_10_plots/TINYIMAGENET_"+"TASKMODEL_"+str(task_model)+"_" + arch + "_MAS_LaMBDA" + str(mas_reg_lambda),
            #                   ylim=(0, 68), bar_widthT=0.12, bar_width=0.14, legend="best")

if 1:

    #===========ResNet

    FINE_RESNET = torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.01b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")
    LwF_RESNET = torch.load("continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.01b1_Falsearch_ResNetdevice_cuda:0regularization_method_LwFshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")

    MAS_RESNET= torch.load("continuallearning_res/num_epochs_100reg_lambda_4.0lr_0.01b1_Falsearch_ResNetdevice_cuda:0regularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")
    results={}
    results["LwF"] = LwF_RESNET["mean_acc"]
    results["MAS"]= MAS_RESNET["mean_acc"]
    results["Finetune"]= FINE_RESNET["mean_acc"]

    hatches=["","",""]
    colors=['C0','C2','C4']
    labels = list(results.keys())  # ['ER Buffer 9k', 'ER Buffer 4.5k', 'ER Buffer 1.8k']
    keys = list(results.keys())
    plot_multibar(results,keys, colors, labels, hatches, save_img="plots/cl_results/TINYIMAGENET_ResNet",ylim=(0, 80),bar_widthT=barwidthT, bar_width=barwidth,legend="best")
    #=======VGG

    FINE_RESNET = torch.load(
        "continuallearning_res/num_epochs_100reg_lambda_0.0lr_0.01b1_Falsearch_VGGdevice_cuda:0r"
        "egularization_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")
    LwF_RESNET = torch.load(
        "continuallearning_res/num_epochs_100reg_lambda_0.1lr_0.01b1_Falsearch_VGGdevice_cuda:0regularization_method_LwFshared_"
        "head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")

    MAS_RESNET= torch.load("continuallearning_res/num_epochs_100reg_lambda_4.0lr_0.01b1_Falsearch_VGGdevice_cuda:0regularization"
                           "_method_MASshared_head_Falseall_initialized_Falsebuffer_size_0batch_size_64lr_decay_rate_0.1dropr_0.0lr_scheduler_plt.pth")
    results={}
    results["LwF"] = LwF_RESNET["mean_acc"]
    results["MAS"]= MAS_RESNET["mean_acc"]
    results["Finetune"]= FINE_RESNET["mean_acc"]

    hatches=["","",""]
    colors=['C0','C2','C4']
    labels = list(results.keys())  # ['ER Buffer 9k', 'ER Buffer 4.5k', 'ER Buffer 1.8k']
    keys = list(results.keys())


    plot_multibar(results,keys, colors, labels, hatches, save_img="plots/cl_results/TINYIMAGENET_VGG",ylim=(0, 80),bar_widthT=barwidthT, bar_width=barwidth,legend="best")
