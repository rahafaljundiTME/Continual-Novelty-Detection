import glob
import torch
print_all_values=1
from Test_Utils import *
reg_lambdas=[1]
novelty_methods=["Gen_based","Mahalanobis","Naive_Threshold","ODIN"]
z_sizes=[8,16,32,63,128]
novelty_batchsizes=[32,64,128,256]
novelty_layer_indes=[0,1,2,3,4]
novelty_magnitudes=[0,0.1,0.002,0.0014,0.001,0.005]
novelty_n_epochs=[10,20,30,50,100]
main_result_file= "Resultstinyimagenet_expnovelty_method"
best_results={}
for arch in ["VGG"]:
    for method in ["MAS"]:
        for reg_lambda in reg_lambdas:


            for novelty_method in novelty_methods:
                main_result_file = "Resultstinyimagenet_exp"
                main_result_file += "*novelty_method_" + str(novelty_method)
                filename = "Results/backup/" + filename + "*.pth"
                print(filename)
                files_path = glob.glob(filename, recursive=True)
                print(files_path)
                max_out_pr=0
                max_results_file=""
                best_results=None
                for file in files_path:
                    results = torch.load(file)
                    out_pr=0
                    for result in results:
                        out_pr+=result.out_data_tpr
                    if max_out_pr<out_pr:
                        max_results_file=file
                        best_results=results
                        max_out_pr=out_pr

                best_results[novelty_method]=best_results
            import pdb;pdb.set_trace()
            #detection errors
            avg_det_errors={key:[] for key in novelty_methods}
            false_p_errs={key:[] for key in novelty_methods}
            false_n_errs = {key: [] for key in novelty_methods}
            for novelty_method in novelty_methods:
                results=best_results[novelty_method]
                for task in range(len(results)):
                    this_avg_det_errors=[]
                    for inset_task_index in range(task):
                        this_avg_det_errors.append(sum(results[task].det_errs[inset_task_index])/len(results[task].det_errs[inset_task_index]))

                    avg_det_errors[novelty_method].append(sum(this_avg_det_errors)/len(this_avg_det_errors))
                    false_n_errs[novelty_method].append(sum(results[task].false_n_errs)/len(results[task].false_n_errs))
                    false_p_errs[novelty_method].append(sum(results[task].false_p_errs)/len(results[task].false_p_errs))


        hatches = ["", "", "",""]
        colors = ['C0', 'C2', 'C4',"C1"]
        labels =novelty_methods
        keys = novelty_methods
        plot_multibar(avg_det_errors, keys, colors, labels, hatches,
                      save_img="TINYIMAGENET_" + arch + "Deterrors" , ylim=(0, 100),
                      bar_widthT=0.12, bar_width=0.14, legend="best")
        plot_multibar(false_p_errs, keys, colors, labels, hatches,
                      save_img="TINYIMAGENET_" + arch + "forgottenAsIn" , ylim=(0, 100),
                      bar_widthT=0.12, bar_width=0.14, legend="best")
        plot_multibar(false_n_errs, keys, colors, labels, hatches,
                      save_img="TINYIMAGENET_" + arch + "PreviousInasOut" , ylim=(0, 100),
                      bar_widthT=0.12, bar_width=0.14, legend="best")

