import glob
import torch
"""
A depricated file, it was used as a draft to compare the performance of different Open_Sim alternatives.
"""
print_all_values = 1
from Test_Utils import *
reg_lambdas = [0, 4]  # [0,1,2,4,8]
novelty_methods = ["Gen_based", "Mahalanobis", "Max_Softmax", "ODIN"]
#novelty_methods = ["Baseline","Max_Softmax"]#
z_sizes = [8, 16, 32, 63, 128]
novelty_batchsizes = [32, 64, 128, 256]
novelty_layer_indes = [0, 1, 2, 3, 4]
novelty_magnitudes = [0, 0.01, 0.002, 0.0014, 0.001, 0.0005]
novelty_n_epochs = [10, 20, 30, 50, 100]
buffer_sizes = [0, 1800, 4500,9000]#0
main_result_file = "Resultstinyimagenet_expnovelty_method"


tune_baed_on = "out"  # "firs_det_errs"
for arch in ["ResNet"]:  # ["VGG","ResNet"]:
    for buffer_size in buffer_sizes:
        for method in ["MAS", "LwF"]:


            for reg_lambda in reg_lambdas:

                if method == "LwF" and reg_lambda > 0:
                    continue
                best_results = {}
                for novelty_method in novelty_methods:



                    main_result_file += "*novelty_method_" + str(novelty_method) + "*" + "_" + str(
                        buffer_size) + "_" + str(arch) + "_" + str(method) + "_" + str(reg_lambda)
                    filename = "" + main_result_file + "*.pth"

                    files_path = glob.glob(filename, recursive=True)
                    # print(files_path)
                    min_out_pr = 1
                    min_results_file = ""
                    this_best_results = None

                    for file in files_path:

                        results = torch.load(file)
                        if len(results) < 10:
                            continue
                        out_pr = 0
                        # for result in results:# this can be reverted to only the first task
                        # only on the first task
                        result = results[0]
                        if tune_baed_on == "out":
                            if result.out_data_tpr == 0:
                                out_pr = 1
                                break
                            out_pr += result.out_data_auc  # /len(results)
                        else:

                            out_pr += result.det_errs[0][0]
                        if min_out_pr > out_pr and out_pr != 0:
                            # import pdb;pdb.set_trace()
                            min_results_file = file
                            this_best_results = results
                            min_out_pr = out_pr

                            print(min_results_file,novelty_method, out_pr)

                    if this_best_results is not None:
                        best_results[novelty_method] = this_best_results
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
                                #print(inset_task_index)
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
                                #import pdb;pdb.set_trace()
                                combined_auc[novelty_method].append( best_results[novelty_method][task_model].combined_auc)
                                combined_det_err[novelty_method].append(best_results[novelty_method][task_model].combined_det_err)

                            if task_model > 0:
                                this_forg_in_errs = [(results[task_model].false_n_errs[i] + results[task_model].false_p_errs[i])/2 for i in range(len(results[task_model].false_n_errs))]
                                this_forg_out_errs = [
                                    (results[task_model].false_out_n_errs[i] + results[task_model].false_out_p_errs[i]) / 2 for i in
                                    range(len(results[task_model].false_out_n_errs))]
                                forg_in_errs[novelty_method].append(
                                    sum(this_forg_in_errs) / len(this_forg_in_errs))
                                if task_model<(len(results)-1):
                                    forg_out_errs[novelty_method].append(
                                        sum(this_forg_out_errs) / len(this_forg_out_errs))

                                combined_forg_in_errs[novelty_method].append((best_results[novelty_method][task_model].combined_false_n_errs +
                                                              best_results[novelty_method][task_model].combined_false_p_errs)/2)



                    hatches = ["", "", "", ""]
                    colors = ['C0', 'C2', 'C4', "C1"]
                    labels = list(best_results.keys())
                    keys = list(best_results.keys())
                    import os
                    if not os.path.exists("plots"):
                        #os.makedirs(directory)
                        os.makedirs("plots/")
                    this_name = "plots/Open_Sim/"+special_char+"/TINYIMAGENET_" + arch + "_" + method + "_" + str(reg_lambda) + "_"+str(buffer_size)+"_"
                    """

                    """

                    print("###############################################################")
                    print(special_char)
                    print(this_name)
                    mult=100
                    import pdb;pdb.set_trace()
                    for key in  keys:

                        printed=""
                        mean_avg_pretask_auc=sum(avg_pretasks_auc[key])/(len(avg_pretasks_auc[key])-1)
                        printed += str(round(mean_avg_pretask_auc * mult, 1)) + " & "
                        mean_avg_current_auc= sum(avg_current_auc[key]) / len(avg_current_auc[key])
                        printed += str(round(mean_avg_current_auc * mult, 1)) + " & "
                        mean_combined_auc = sum(combined_auc[key]) / len(combined_auc[key])
                        printed += str(round(mean_combined_auc * mult, 1)) + " & "
                        mean_avg_auc = sum(avg_auc[key]) / len(avg_auc[key])
                        printed += str(round(mean_avg_auc * mult, 1)) + " & "
                        mean_combined_det_err= sum(combined_det_err[key]) / len(combined_det_err[key])
                        printed += str(round(mean_combined_det_err * mult, 1)) + " & "
                        mean_avg_forg_in=sum(forg_in_errs[key]) / len(forg_in_errs[key])
                        printed += str(round(mean_avg_forg_in * mult, 1)) + " & "
                        mean_avg_forg_out = sum(forg_out_errs[key]) / len(forg_out_errs[key])
                        printed+=str(round(mean_avg_forg_out*mult,1))
                        print(key, "&", printed)
#                     try:


#                         #import pdb;pdb.set_trace()
#                         barwidthT=0.18
#                         barwidth=0.20
#                         plot_multibar(avg_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "Deterrors", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="Detection Errors")
#                         plot_multibar(avg_pretasks_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousTasks_Deterrors", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="Detection Errors")
#                         plot_multibar(avg_current_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "CurrentTask_Deterrors", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="Detection Errors")


#                         plot_multibar(avg_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "AUCs", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                         plot_multibar(avg_pretasks_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousTasks_AUCs", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                         plot_multibar(avg_current_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "CurrentTask_AUCs", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")



#                         plot_multibar(forg_in_errs, keys, colors, labels, hatches,
#                                       save_img=this_name+ "forgotten_vs_in", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="Detection Errors")
#                         plot_multibar(forg_out_errs, keys, colors, labels, hatches,
#                                       save_img=this_name+ "Forgotten_vs_out", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="Detection Errors")

#                         plot_multibar(combined_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "Combined_AUCs", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                         plot_multibar(combined_det_err, keys, colors, labels, hatches,
#                                       save_img=this_name + "Combined_Det_errs", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                         plot_multibar(combined_forg_in_errs, keys, colors, labels, hatches,
#                                       save_img=this_name + "Combined_Forg_in", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                         plot_multibar(combined_forg_out_errs, keys, colors, labels, hatches,
#                                       save_img=this_name + "Combined_Forg_out", ylim=(0, 1),
#                                       bar_widthT=barwidthT, bar_width=barwidth, legend="best",ylabel="AUC")
#                     except:
#                         print(this_name+" is not available!")
#                         pass

#                     # detection errors
#                     avg_det_errors = {key: [] for key in novelty_methods}
#                     avg_auc = {key: [] for key in novelty_methods}
#                     avg_pretasks_det_errors = {key: [] for key in novelty_methods}
#                     avg_current_det_errors = {key: [] for key in novelty_methods}
#                     avg_pretasks_auc = {key: [] for key in novelty_methods}
#                     avg_current_auc = {key: [] for key in novelty_methods}
#                     false_p_errs = {key: [] for key in novelty_methods}
#                     false_n_errs = {key: [] for key in novelty_methods}
#                     false_againtcurrent_n_errs = {key: [] for key in novelty_methods}
#                     for novelty_method in best_results.keys():
#                         results = best_results[novelty_method]

#                         #print(novelty_method)

#                         for task_model in range(len(results)):
#                             this_avg_det_errors = []
#                             this_avg_pre_det_errors = []
#                             this_avg_current_det_errors = []
#                             this_avg_auc = []
#                             this_avg_pre_auc = []
#                             this_avg_current_auc = []
#                             for inset_task_index in range(task_model + 1):
#                                 #print(inset_task_index)
#                                 if len(results[task_model].det_errs[inset_task_index]) > 0:
#                                     this_task_avg_det = sum(results[task_model].det_errs[inset_task_index]) / len(
#                                         results[task_model].det_errs[inset_task_index])
#                                     this_task_avg_auc = sum(results[task_model].auc[inset_task_index]) / len(
#                                         results[task_model].auc[inset_task_index])
#                                     this_avg_det_errors.append(this_task_avg_det)
#                                     this_avg_auc.append(this_task_avg_auc)
#                                     if inset_task_index < task_model:
#                                         # previosly seen error
#                                         this_avg_pre_det_errors.append(this_task_avg_det)
#                                         this_avg_pre_auc.append(this_task_avg_auc)
#                                     else:
#                                         this_avg_current_det_errors.append(this_task_avg_det)
#                                         this_avg_current_auc.append(this_task_avg_auc)

#                             if len(this_avg_det_errors) > 0:
#                                 avg_det_errors[novelty_method].append(
#                                     sum(this_avg_det_errors) / len(this_avg_det_errors))
#                                 if len(this_avg_pre_det_errors) > 0:  # not the first task
#                                     avg_pretasks_det_errors[novelty_method].append(
#                                         sum(this_avg_pre_det_errors) / len(this_avg_pre_det_errors))
#                                     avg_pretasks_auc[novelty_method].append(
#                                         sum(this_avg_pre_auc) / len(this_avg_pre_auc))
#                                 avg_current_det_errors[novelty_method].append(
#                                     sum(this_avg_current_det_errors) / len(this_avg_current_det_errors))
#                                 avg_auc[novelty_method].append(sum(this_avg_auc) / len(this_avg_auc))
#                                 avg_current_auc[novelty_method].append(
#                                     sum(this_avg_current_auc) / len(this_avg_current_auc))
#                                 if task_model == 0:
#                                     false_n_errs[novelty_method].append(0)
#                                     false_p_errs[novelty_method].append(0)
#                                 # if task_model > 0:
#                                 # import pdb;pdb.set_trace()
#                                 # false_againtcurrent_n_errs[novelty_method].append(sum(results[task_model].false_againtcurrent_n_errs) \
#                                 #                                                  / len(results[task_model].false_againtcurrent_n_errs))

#                             if task_model > 0:
#                                 false_n_errs[novelty_method].append(
#                                     sum(results[task_model].false_n_errs) / len(results[task_model].false_n_errs))
#                                 false_p_errs[novelty_method].append(
#                                     sum(results[task_model].false_p_errs) / len(results[task_model].false_p_errs))

#                     hatches = ["", "", "", ""]
#                     colors = ['C0', 'C2', 'C4', "C1"]
#                     labels = list(best_results.keys())
#                     keys = list(best_results.keys())
#                     import os
#                     if not os.path.exists("plots/Open_Sim/"+special_char):
#                         #os.makedirs(directory)
#                         os.makedirs("plots/Open_Sim/"+special_char)
#                     this_name = "plots/Open_Sim/"+special_char+"/TINYIMAGENET_" + arch + "_" + method + "_" + str(reg_lambda) + "_"+str(buffer_size)+"_"
#                     #import pdb;pdb.set_trace()
#                     try:

#                         plot_multibar(avg_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "Deterrors", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")
#                         plot_multibar(avg_pretasks_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousTasks_Deterrors", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")
#                         plot_multibar(avg_current_det_errors, keys, colors, labels, hatches,
#                                       save_img=this_name + "CurrentTask_Deterrors", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")

#                         plot_multibar(avg_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "AUCs", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="AUC")
#                         plot_multibar(avg_pretasks_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousTasks_AUCs", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="AUC")
#                         plot_multibar(avg_current_auc, keys, colors, labels, hatches,
#                                       save_img=this_name + "CurrentTask_AUCs", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="AUC")

#                         plot_multibar(false_p_errs, keys, colors, labels, hatches,
#                                       save_img=this_name + "forgottenAsIn", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")
#                         plot_multibar(false_n_errs, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousInasOut", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")
#                         plot_multibar(false_againtcurrent_n_errs, keys, colors, labels, hatches,
#                                       save_img=this_name + "PreviousInasOut_based_onCurrent", ylim=(0, 1),
#                                       bar_widthT=0.12, bar_width=0.14, legend="best", ylabel="Detection Errors")
#                     except:
#                         print(this_name + " is not available!")
#                         pass
