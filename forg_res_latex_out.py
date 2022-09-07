import glob
import torch

print_all_values = 1
from Test_Utils import *

reg_lambdas =[0.0,4.0]#[0.0,20.0]# [20.0,1.0,2.0,4.0,8.0]
novelty_methods = ["Gen_based", "Mahalanobis", "Max_Softmax", "ODIN"]
z_sizes = [8, 16, 32, 63, 128]
novelty_batchsizes = [32, 64, 128, 256]
novelty_layer_indes = [0, 1, 2, 3, 4]
novelty_magnitudes = [0, 0.01, 0.002, 0.0014, 0.001, 0.0005]
novelty_n_epochs = [10, 20, 30, 50, 100]
main_result_file = "Resultstinyimagenet_expnovelty_method"

tune_baed_on="out"#"firs_det_errs"
archs=["ResNet"]#["ResNet","VGG"] #tiny Imagenet option
#main_name="Resultstinyimagenet_exp"
main_name="./novelty_results/8tasks/final/Results8tasks_exp"
#main_name="./forg_wise_nov_results/8tasks/Results8tasks_exp"
main_name="./forg_wise_nov_results/multi_head/Resultstinyimagenet_exp"
#main_name="./forg_wise_nov_results/shared_head/Resultstinyimagenet_exp"
#main_name="./novelty_results/Tiny_imagenet/Results_multihead/final/Resultstinyimagenet_exp"
#main_name="./novelty_results/Tiny_imagenet/Results_sharedhead/final/Resultstinyimagenet_exp"
output_name="8tasks/"
output_name="TinyImageNet/multihead/ResNet"

#output_name="TinyImageNet/multihead/VGG"
#output_name="TinyImageNet/sharedhead/ResNet"
#archs=["VGG"]#["ResNet"]#["Alex"]
archs=["ResNet"]
buffer_sizes =[0]#[1800, 4500, 9000]
tune_baed_on="out"#"firs_det_errs"
for arch in archs:
    for method in ["MAS","LwF"]:
        for reg_lambda in reg_lambdas:
            if method=="LwF" and reg_lambda>0:
                continue
            for buffer_size in buffer_sizes:
                best_results={}
                opts=Options()
                opts.buffer_size=buffer_size
                opts.main_result_file =main_name
                opts.arch=arch
                opts.method=method
                opts.reg_lambda=reg_lambda
                opts.tune_baed_on=tune_baed_on
                best_results=get_methods_best_results(novelty_methods, opts)
                # detection errors
                avg_det_errors = {key: [] for key in novelty_methods}
                avg_auc = {key: [] for key in novelty_methods}
                avg_pretasks_det_errors = {key: [] for key in novelty_methods}
                avg_current_det_errors = {key: [] for key in novelty_methods}
                avg_pretasks_auc = {key: [] for key in novelty_methods}
                avg_current_auc = {key: [] for key in novelty_methods}
                forg_in_errs = {key: [] for key in novelty_methods}
                forg_out_errs = {key: [] for key in novelty_methods}
                in_forg_auc= {key: [] for key in novelty_methods}
                out_forg_auc= {key: [] for key in novelty_methods}
                false_againtcurrent_n_errs= {key: [] for key in novelty_methods}
                combined_auc = {key: [] for key in novelty_methods}
                combined_det_err = {key: [] for key in novelty_methods}
                combined_forg_in_errs = {key: [] for key in novelty_methods}
                combined_forg_out_errs= {key: [] for key in novelty_methods}
                for novelty_method in best_results.keys():
                    results = best_results[novelty_method]

                    print(novelty_method)

                    for task_model in range(len(results)):
                        this_avg_det_errors = []
                        this_avg_pre_det_errors=[]
                        this_avg_current_det_errors = []
                        this_avg_auc = []
                        this_avg_pre_auc=[]
                        this_avg_current_auc = []
                        for inset_task_index in range(task_model + 1):
                            print(inset_task_index)
                            if len(results[task_model].det_errs[inset_task_index]) > 0:
                                this_task_avg_det= sum(results[task_model].det_errs[inset_task_index]) / len(
                                    results[task_model].det_errs[inset_task_index]) #average detection err for a given in task
                                this_task_avg_auc = sum(results[task_model].auc[inset_task_index]) / len(
                                    results[task_model].auc[inset_task_index])
                                this_avg_det_errors.append(this_task_avg_det)
                                this_avg_auc.append(this_task_avg_auc)
                                if inset_task_index<task_model:
                                    #previosly seen error
                                    this_avg_pre_det_errors.append(this_task_avg_det)
                                    this_avg_pre_auc.append(this_task_avg_auc)
                                # else:
                                #     this_avg_current_det_errors.append(this_task_avg_det)
                                #     this_avg_current_auc.append(this_task_avg_auc)

                        if len(this_avg_det_errors) > 0:
                            avg_det_errors[novelty_method].append(sum(this_avg_det_errors) / len(this_avg_det_errors))#all_average for a model
                            avg_auc[novelty_method].append(sum(this_avg_auc) / len(this_avg_auc))  # check!
                            if len(this_avg_pre_det_errors)>0:#not the first task
                                avg_pretasks_det_errors[novelty_method].append(sum(this_avg_pre_det_errors) / len(this_avg_pre_det_errors))#all pre average for a model
                                avg_pretasks_auc[novelty_method].append(sum(this_avg_pre_auc) / len(this_avg_pre_auc))
                            else:
                                avg_pretasks_det_errors[novelty_method].append(0)
                                avg_pretasks_auc[novelty_method].append(0)

                            avg_current_det_errors[novelty_method].append(this_task_avg_det)

                            avg_current_auc[novelty_method].append(this_task_avg_auc)
                            if task_model == 0:
                                forg_in_errs[novelty_method].append(0)
                                forg_out_errs[novelty_method].append(0)
                                in_forg_auc[novelty_method].append(0)
                                out_forg_auc[novelty_method].append(0)
                            combined_auc[novelty_method].append( best_results[novelty_method][task_model].combined_auc)
                            combined_det_err[novelty_method].append(best_results[novelty_method][task_model].combined_det_err)

                        if task_model > 0:
                            this_forg_in_errs = [(results[task_model].false_n_errs[i] + results[task_model].false_p_errs[i])/2 for i in range(len(results[task_model].false_n_errs))]
                            this_forg_out_errs = [
                                (results[task_model].false_out_n_errs[i] + results[task_model].false_out_p_errs[i]) / 2 for i in
                                range(len(results[task_model].false_out_n_errs))]
                            forg_in_errs[novelty_method].append(
                                sum(this_forg_in_errs) / len(this_forg_in_errs))
                            in_forg_auc[novelty_method].append(sum(results[task_model].in_forg_auc) / len(results[task_model].in_forg_auc))
                            if task_model<(len(results)-1):
                                out_forg_auc[novelty_method].append(sum(results[task_model].out_forg_auc) / len(results[task_model].out_forg_auc))
                                forg_out_errs[novelty_method].append(
                                    sum(this_forg_out_errs) / len(this_forg_out_errs))

                            combined_forg_in_errs[novelty_method].append((best_results[novelty_method][task_model].combined_false_n_errs +
                                                          best_results[novelty_method][task_model].combined_false_p_errs)/2)



                hatches = ["", "", "", ""]
                colors = ['C0', 'C2', 'C4', "C1"]
                labels = {"Max_Softmax":"nav","ODIN":"odin","Mahalanobis":"mah","Gen_based":"vae"}
                keys = [ "Max_Softmax", "ODIN", "Mahalanobis","Gen_based"]
                this_name = "plots/"+output_name+"/" + arch + "_" + method + "_" + str(reg_lambda) + "_"+str(buffer_size)+"_"
                """
                avg_det_errors = {key: [] for key in novelty_methods}
                avg_auc = {key: [] for key in novelty_methods}
                avg_pretasks_det_errors = {key: [] for key in novelty_methods}
                avg_current_det_errors = {key: [] for key in novelty_methods}
                avg_pretasks_auc = {key: [] for key in novelty_methods}
                avg_current_auc = {key: [] for key in novelty_methods}
                forg_in_errs = {key: [] for key in novelty_methods}
                forg_out_errs = {key: [] for key in novelty_methods}
                false_againtcurrent_n_errs= {key: [] for key in novelty_methods}
                combined_auc = {key: [] for key in novelty_methods}
                combined_det_err = {key: [] for key in novelty_methods}
                combined_forg_in_errs = {key: [] for key in novelty_methods}
                combined_forg_out_errs= {key: [] for key in novelty_methods}
                """
                #forg_in_errs
                #forg_out_errs
                mult=100

                try:
                    for key in  keys:
                        printed=""
                        for i in range(1,len(out_forg_auc[key])):
                            printed+=str(round(in_forg_auc[key][i]*mult,1))+" & "+str(round(out_forg_auc[key][i]*mult,1))+"&"
                        printed+=str(round(in_forg_auc[key][i+1]*mult,1))+" & "+"- "
                        print(labels[key],"&",printed)
                except:
                    print(this_name+" is not available!")
                    pass
                print("*****************************************************************")
                try:
                    for key in  keys:
                        printed=""
                        for i in range(1,len(forg_out_errs[key])):
                            printed+=str(round(forg_in_errs[key][i]*mult,1))+" & "+str(round(forg_out_errs[key][i]*mult,1))+"&"
                        printed+=str(round(forg_in_errs[key][i+1]*mult,1))+" & "+"- "
                        print(labels[key],"&",printed)
                except:
                    print(this_name+" is not available!")
                    pass
