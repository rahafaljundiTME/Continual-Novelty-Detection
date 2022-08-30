"""
        self.features_norm_bef = []
        self.features_norm_aft = []
        self.cos_softmax=[]#biased_prod_max
        self.used_delta=[]#biased_prod_std
        self.prod_max=[]#prod_max
        self.prod_std= []#prod_std
        self.org_sotmax=[]#prod_aft_max
        self.mean_dot_product=[]
        self.delta=[]
"""
"""
generates the main stats (mean, norm, std) of the IN/OUT/Forgotten sets at different stages in the sequence.
"""
from Test_Utils import *
import operator
def print_key_info(key,nvmodel,task,bins=None):
    flag=True
    while flag and task>-1:
        try:
            #print("==============" + key + "==================")
            #import pdb;pdb.set_trace()
            features_norm_bef = np.mean(nvmodel.features_norm_bef[task][key])
    
            features_mean = np.mean(nvmodel.features_mean[task][key])
            features_std = np.mean(nvmodel.features_std[task][key])
            cos_softmax = np.mean(np.array(nvmodel.cos_softmax[task][key]),axis=0)
            org_sotmax = np.mean(np.array(nvmodel.org_sotmax[task][key]),axis=0)
            prod_max = np.mean(nvmodel.prod_max[task][key])
            prod_std = np.mean(nvmodel.prod_std[task][key])
            z = 1/(1 + np.exp(-20*np.array(nvmodel.used_delta[task][key])))
            used_delta = np.mean(z)
            delta = np.mean(nvmodel.delta[task][key])
            #mean_dot_product = np.mean(nvmodel.mean_dot_product[task][key])
            print("==============" + key + "==================")
#             print("features_norm_bef", features_norm_bef, "features_norm_aft", features_norm_aft,"prod_max", prod_max, "prod_std", prod_std, "cos_softmax",cos_softmax,"org_sotmax",org_sotmax,"dot_product_mean",mean_dot_product,
#                  "used_delta", used_delta, "delta", delta)
            #print( "cos_softmax",cos_softmax)
            #print("org_sotmax",org_sotmax)
            #if bins is None:
           
            #x=nvmodel.features_norm_std[task][key]
            #normhist,normbin=np.histogram(z)   
            #else:
            #    normhist,normbin=np.histogram(nvmodel.used_delta[task][key],bins=bins)
            #print("used_delta_hist",normhist)
            #print("used_delta_bin",normbin)


            #softhist,softbin=np.histogram(nvmodel.org_sotmax[task][key]) 
            #print("org_sotmax_hist",softhist)
            #print("org_sotmax_bin",softbin)       
            flag=False
            #return features_norm_bef
            return features_norm_bef,features_mean,features_std
        except:
            task-=1
"""
['in_val', 'out_val', 'in0_0', 'out0_0_1', 'out0_0_2', 'out0_0_3', 'out0_0_4', 'out0_0_5', 'out0_0_6', 'out0_0_7',
 'out0_0_8', 'out0_0_9', 'in1_0', 'out1_0_1', 'out1_0_2', 'out1_0_3', 'out1_0_4', 'out1_0_5', 'out1_0_6', 'out1_0_7',
  'out1_0_8', 'Forgotten1_0', 'in2_0', 'out2_0_1', 'out2_0_2', 'out2_0_3', 'out2_0_4', 'out2_0_5', 'out2_0_6', 'out2_0_7', 'Forgotten2_0', 'in3_0', 'out3_0_1', 'out3_0_2', 'out3_0_3', 'out3_0_4', 'out3_0_5', 'out3_0_6', 'Forgotten3_0', 'in4_0', 'out4_0_1', 'out4_0_2', 'out4_0_3', 'out4_0_4', 'out4_0_5', 'Forgotten4_0', 'in5_0', 'out5_0_1', 'out5_0_2', 'out5_0_3', 'out5_0_4', 'Forgotten5_0', 'in6_0', 'out6_0_1', 'out6_0_2', 'out6_0_3', 'Forgotten6_0', 'in7_0', 'out7_0_1', 'out7_0_2', 'Forgotten7_0', 'in8_0', 'out8_0_1', 'Forgotten8_0', 'in9_0', 'Forgotten9_0']
"""
import torch
import numpy as np
nvmodels_names=["NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_1800_ResNet_MAS_0.0.pth","NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_4500_ResNet_MAS_0.0.pth","NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_9000_ResNet_MAS_0.0.pth","NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_0_ResNet_MAS_0.0.pth","NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_0_ResNet_MAS_4.0.pth","NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_0_ResNet_LwF_0.0.pth"]
this_names=["stats/shared_head_ER1800","stats/shared_head_ER4500","stats/shared_head_ER9000","stats/multi_head_fine","stats/multi_head_MAS","stats/multi_head_LwF"]

for i in range(len(nvmodels_names)):
    
    nvmodel=torch.load(nvmodels_names[i])
    this_name=this_names[i]
    first_in_norm=[]
    out_norm=[]
    pre_in_norm=[]
    last_in_norm=[]
    forg_norm=[]
    first_in_mean=[]
    out_mean=[]
    pre_in_mean=[]
    last_in_mean=[]
    forg_mean=[]
    first_in_std=[]
    out_std=[]
    pre_in_std=[]
    last_in_std=[]
    forg_std=[]
    for task in range(0,len(nvmodel.features_norm_bef)):
        print("Model of task:", task)
        #import pdb;pdb.set_trace()
        middle_norm=0
        middle_mean=0
        middle_std=0
        for inset_task_index in range(task+1):
            print("inset task:", inset_task_index)
            key="in"+str(task)+"_"+str(inset_task_index)
            #features_norm_bef,features_mean,features_std
            in_norm,in_mean,in_std=print_key_info(key, nvmodel, task)
            if inset_task_index==0:
                first_in_norm.append(in_norm)
                first_in_mean.append(in_mean)
                first_in_std.append(in_std)
            if inset_task_index==task:
                last_in_norm.append(in_norm)
                last_in_mean.append(in_mean)
                last_in_std.append(in_std)
            if inset_task_index>0 and inset_task_index<task:
                middle_norm+=in_norm
                middle_mean+=in_mean 
                middle_std+=in_std
            #import pdb;pdb.set_trace()
            if inset_task_index==0:
                this_out_norm=0
                this_out_mean=0
                this_out_std=0
                named_out=0
                for out_task in range(task+1,10):#len(nvmodel.features_norm_bef[inset_task_index])):
                    named_out+=1
                    key = "out" + str(task) + "_" + str(inset_task_index)+ "_" +str(named_out )
                    #print("out_task", key)  

                    norm,mean,std=print_key_info(key, nvmodel, task)
                    this_out_norm+=norm
                    this_out_mean+=mean
                    this_out_std+=std
                if named_out>0:
                    this_out_norm=this_out_norm/named_out
                    this_out_mean=this_out_mean/named_out
                    this_out_std=this_out_std/named_out
                out_norm.append(this_out_norm)
                out_mean.append(this_out_mean)
                out_std.append(this_out_std)
        this_forg_norm=0
        this_forg_mean=0
        this_forg_std=0
        for fogotten_task in range(task):
            key = "Forgotten"  + str(task) + "_" + str(fogotten_task)
            in_norm,in_mean,in_std=print_key_info(key, nvmodel, task)
            this_forg_norm+=in_norm/task
            this_forg_mean+=in_mean/task
            this_forg_std+=in_std/task
        forg_norm.append(this_forg_norm)   
        forg_mean.append(this_forg_mean) 
        forg_std.append(this_forg_std) 
        divider=1 if task<2 else  task-1
        pre_in_norm.append(middle_norm/(divider)) 
        pre_in_mean.append(middle_mean/(divider))
        pre_in_std.append(middle_std/(divider))
        #==========================plot============================
    barwidthT=0.16
    barwidth=0.18

    all_feat_norm = {}
    all_feat_norm["first_in"]=first_in_norm
    all_feat_norm["pre_in"]=pre_in_norm
    all_feat_norm["last_in"]=last_in_norm
    all_feat_norm["out"]=out_norm
    all_feat_norm["forg"]=forg_norm
    maxval=max([max(first_in_norm),max(pre_in_norm),max(last_in_norm),max(out_norm),max(forg_norm)])
    minval=min([max(first_in_norm),min(pre_in_norm),min(last_in_norm),min(out_norm),max(forg_norm)])
    hatches = ["", "", "", "",""]
    legend="best"
    colors = ['C1', 'C3', 'C2', "C6","C9"]
    #key_labels = { "Mahalanobis":"Mahalanobis","ODIN":"ODIN", "Max_Softmax":"Softmax","Gen_based":"VAE"}
    handle="FeatureNorm"

    keys =list(all_feat_norm.keys())#list(best_results.keys())

    labels = keys#[key_labels[key] for key in keys]
    plot_multibar(all_feat_norm, keys, colors, labels, hatches,
                  save_img=this_name + handle, ylim=(minval,maxval+maxval/10),
                  bar_widthT=barwidthT, bar_width=barwidth, legend=legend,ylabel=handle)

    #=========================================================
   
    all_feat_norm = {}
    all_feat_norm["first_in"]=first_in_mean
    all_feat_norm["pre_in"]=pre_in_mean
    all_feat_norm["last_in"]=last_in_mean
    all_feat_norm["out"]=out_mean
    all_feat_norm["forg"]=forg_mean
    maxval=max([max(first_in_mean),max(pre_in_mean),max(last_in_mean),max(out_mean),max(forg_mean)])
    minval=min([max(first_in_mean),min(pre_in_mean),min(last_in_mean),min(out_mean),max(forg_mean)])
    hatches = ["", "", "", "",""]
    legend="best"
    colors = ['C1', 'C3', 'C2', "C6","C9"]
    #key_labels = { "Mahalanobis":"Mahalanobis","ODIN":"ODIN", "Max_Softmax":"Softmax","Gen_based":"VAE"}

    handle="FeatureMean"
    keys =list(all_feat_norm.keys())#list(best_results.keys())

    labels = keys#[key_labels[key] for key in keys]
    plot_multibar(all_feat_norm, keys, colors, labels, hatches,
                  save_img=this_name + handle, ylim=(minval,maxval+maxval/10),
                  bar_widthT=barwidthT, bar_width=barwidth, legend=legend,ylabel=handle)
    #===================================================================================
    all_feat_norm = {}
    all_feat_norm["first_in"]=first_in_std
    all_feat_norm["pre_in"]=pre_in_std
    all_feat_norm["last_in"]=last_in_std
    all_feat_norm["out"]=out_std
    all_feat_norm["forg"]=forg_std
    maxval=max([max(first_in_std),max(pre_in_std),max(last_in_std),max(out_std),max(forg_std)])
    minval=min([max(first_in_std),min(pre_in_std),min(last_in_std),min(out_std),max(forg_std)])
    hatches = ["", "", "", "",""]
    legend="best"
    colors = ['C1', 'C3', 'C2', "C6","C9"]
    #key_labels = { "Mahalanobis":"Mahalanobis","ODIN":"ODIN", "Max_Softmax":"Softmax","Gen_based":"VAE"}
    handle="FeatureStd"
    keys =list(all_feat_norm.keys())#list(best_results.keys())

    labels = keys#[key_labels[key] for key in keys]
    plot_multibar(all_feat_norm, keys, colors, labels, hatches,
                  save_img=this_name + handle, ylim=(minval,maxval+maxval/10),
                  bar_widthT=barwidthT, bar_width=barwidth, legend=legend,ylabel=handle)