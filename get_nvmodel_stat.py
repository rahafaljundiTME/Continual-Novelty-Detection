"""
A depricated file which prints the features stats from Open_Sim model.
"""
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
def print_key_info(key,nvmodel,task,bins=None):
    flag=True
    while flag and task>-1:
        try:
            #print("==============" + key + "==================")
            #import pdb;pdb.set_trace()
            features_norm_bef = np.mean(nvmodel.features_norm_bef[task][key])
            features_norm_aft = np.mean(nvmodel.features_norm_aft[task][key])
            cos_softmax = np.mean(np.array(nvmodel.cos_softmax[task][key]),axis=0)
            org_sotmax = np.mean(np.array(nvmodel.org_sotmax[task][key]),axis=0)
            prod_max = np.mean(nvmodel.prod_max[task][key])
            prod_std = np.mean(nvmodel.prod_std[task][key])
            z = 1/(1 + np.exp(-20*np.array(nvmodel.used_delta[task][key])))
            used_delta = np.mean(z)
            delta = np.mean(nvmodel.delta[task][key])
            mean_dot_product = np.mean(nvmodel.mean_dot_product[task][key])
            print("==============" + key + "==================")
            print("features_norm_bef", features_norm_bef, "features_norm_aft", features_norm_aft,"prod_max", prod_max, "prod_std", prod_std, "cos_softmax",cos_softmax,"org_sotmax",org_sotmax,"dot_product_mean",mean_dot_product,
                 "used_delta", used_delta, "delta", delta)
            #print( "cos_softmax",cos_softmax)
            #print("org_sotmax",org_sotmax)
            #if bins is None:

            normhist,normbin=np.histogram(z)   
            #else:
            #    normhist,normbin=np.histogram(nvmodel.used_delta[task][key],bins=bins)
            print("used_delta_hist",normhist)
            print("used_delta_bin",normbin)


            softhist,softbin=np.histogram(nvmodel.org_sotmax[task][key]) 
            print("org_sotmax_hist",softhist)
            print("org_sotmax_bin",softbin)       
            flag=False
            return normbin
        except:
            task-=1
"""
['in_val', 'out_val', 'in0_0', 'out0_0_1', 'out0_0_2', 'out0_0_3', 'out0_0_4', 'out0_0_5', 'out0_0_6', 'out0_0_7',
 'out0_0_8', 'out0_0_9', 'in1_0', 'out1_0_1', 'out1_0_2', 'out1_0_3', 'out1_0_4', 'out1_0_5', 'out1_0_6', 'out1_0_7',
  'out1_0_8', 'Forgotten1_0', 'in2_0', 'out2_0_1', 'out2_0_2', 'out2_0_3', 'out2_0_4', 'out2_0_5', 'out2_0_6', 'out2_0_7', 'Forgotten2_0', 'in3_0', 'out3_0_1', 'out3_0_2', 'out3_0_3', 'out3_0_4', 'out3_0_5', 'out3_0_6', 'Forgotten3_0', 'in4_0', 'out4_0_1', 'out4_0_2', 'out4_0_3', 'out4_0_4', 'out4_0_5', 'Forgotten4_0', 'in5_0', 'out5_0_1', 'out5_0_2', 'out5_0_3', 'out5_0_4', 'Forgotten5_0', 'in6_0', 'out6_0_1', 'out6_0_2', 'out6_0_3', 'Forgotten6_0', 'in7_0', 'out7_0_1', 'out7_0_2', 'Forgotten7_0', 'in8_0', 'out8_0_1', 'Forgotten8_0', 'in9_0', 'Forgotten9_0']
"""
import torch
import numpy as np

nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_61novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_MAS_0.0.pth"
nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_61novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_MAS_4.0.pth"
nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_61novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_LwF_0.0.pth"
nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_1800_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_4500_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_9000_ResNet_MAS_0.0.pth"
nvmodel_name="Open_Sim_Trials/NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_66novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_LwF_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_66novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_MAS_4.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_66novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_1800_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_4500_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_10.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_9000_ResNet_MAS_0.0.pth"
#nvmodel_name="Open_Sim_Trials/NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_2novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_4500_ResNet_MAS_0.0.pth"
#nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_ResNet_MAS_4.0.pth"
#nvmodel_name="NVMODEL_Results8tasks_expnovelty_method_Open_Sim_perturbnovelty_magnitude_1.0novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_32novelty_max_beta_1novelty_n_epochs_100novelty_tuned_on_in_task_False_0_Alex_MAS_20.0.pth"
nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_1800_ResNet_MAS_0.0.pth"
nvmodel_name="NVMODEL_Resultstinyimagenet_expnovelty_method_Open_Statnovelty_magnitude_1.0novelty_temperature_0.5novelty_batchsize_200novelty_feature_based_Falsenovelty_z_size_1novelty_n_epochs_3novelty_tuned_on_in_task_False_0_ResNet_MAS_0.0.pth"
nvmodel=torch.load(nvmodel_name)


for task in range(0,len(nvmodel.features_norm_bef)):
    print("Model of task:", task)
    import pdb;pdb.set_trace()
    for inset_task_index in range(task+1):
        print("inset task:", inset_task_index)
        key="in"+str(task)+"_"+str(inset_task_index)
        bins=print_key_info(key, nvmodel, task)
        #import pdb;pdb.set_trace()
        named_out=0
        for out_task in range(task+1,10):#len(nvmodel.features_norm_bef[inset_task_index])):
            named_out+=1
            key = "out" + str(task) + "_" + str(inset_task_index)+ "_" +str(named_out )
            #print("out_task", key)          
            print_key_info(key, nvmodel, task,bins)

    for fogotten_task in range(task):
        key = "Forgotten"  + str(task) + "_" + str(fogotten_task)
        print_key_info(key, nvmodel, task)

