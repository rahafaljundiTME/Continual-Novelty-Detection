"""
self.features_norm_bef = []
self.features_norm_aft = []
self.biased_prod_max = []
self.biased_prod_std = []
self.prod_max = []
self.prod_std = []
self.prod_aft_max = []
self.delta = []
"""
"""
A deprecated file which print the stats gathered in Open_Sim_perturb file.
"""
def print_key_info(key,nvmodel,task):
    print("==============" + key + "==================")
    features_norm_bef = np.mean(nvmodel.features_norm_bef[task][key])
    features_norm_aft = np.mean(nvmodel.features_norm_aft[task][key])
    biased_prod_max = np.mean(nvmodel.biased_prod_max[task][key])
    biased_prod_std = np.mean(nvmodel.biased_prod_std[task][key])
    prod_max = np.mean(nvmodel.prod_max[task][key])
    prod_std = np.mean(nvmodel.prod_std[task][key])
    prod_aft_max = np.mean(nvmodel.prod_aft_max[task][key])
    delta = np.mean(nvmodel.delta[task][key])
    print("features_norm_bef", features_norm_bef, "features_norm_aft", features_norm_aft, "biased_prod_max",
          biased_prod_max,
          "biased_prod_std", biased_prod_std, "prod_max", prod_max, "prod_std", prod_std, "prod_aft_max", prod_aft_max,
          "delta", delta)
"""
['in_val', 'out_val', 'in0_0', 'out0_0_1', 'out0_0_2', 'out0_0_3', 'out0_0_4', 'out0_0_5', 'out0_0_6', 'out0_0_7',
 'out0_0_8', 'out0_0_9', 'in1_0', 'out1_0_1', 'out1_0_2', 'out1_0_3', 'out1_0_4', 'out1_0_5', 'out1_0_6', 'out1_0_7',
  'out1_0_8', 'Forgotten1_0', 'in2_0', 'out2_0_1', 'out2_0_2', 'out2_0_3', 'out2_0_4', 'out2_0_5', 'out2_0_6', 'out2_0_7', 'Forgotten2_0', 'in3_0', 'out3_0_1', 'out3_0_2', 'out3_0_3', 'out3_0_4', 'out3_0_5', 'out3_0_6', 'Forgotten3_0', 'in4_0', 'out4_0_1', 'out4_0_2', 'out4_0_3', 'out4_0_4', 'out4_0_5', 'Forgotten4_0', 'in5_0', 'out5_0_1', 'out5_0_2', 'out5_0_3', 'out5_0_4', 'Forgotten5_0', 'in6_0', 'out6_0_1', 'out6_0_2', 'out6_0_3', 'Forgotten6_0', 'in7_0', 'out7_0_1', 'out7_0_2', 'Forgotten7_0', 'in8_0', 'out8_0_1', 'Forgotten8_0', 'in9_0', 'Forgotten9_0']
"""
import torch
import numpy as np

nvmodel_name=""
nvmodel=torch.load(nvmodel_name)

for task in range(len(nvmodel.features_norm_bef)):
    print("Model of task:", task)
    for inset_task_index in range(task+1):
        print("inset task:", inset_task_index)
        key="in"+"_"+str(task)+"_"+str(inset_task_index)
        print_key_info(key, nvmodel, task)

        for out_task in range(task+1,len(nvmodel.features_norm_bef)):
            key = "out" + "_" + str(task) + "_" + str(inset_task_index)+ "_" +str(out_task )

            print_key_info(key, nvmodel, task)

    for fogotten_task in range(task):
        key = "Forgotten" + "_" + str(task) + "_" + str(fogotten_task)
        print_key_info(key, nvmodel, task)

