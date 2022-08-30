import torch
import os
from pydoc import locate
from NoveltyDetectionModules import Metrics,Max_Softmax,ODIN,NvDModules
from NoveltyDetectionModules.NvDModules import NvDResults
from torchvision import datasets, transforms

class CNV_Evaluator(object):
    """An abstract class encapsulating a novelty detection method
    """
    def __init__(self,cn_models,data_sequence,opts):
        self.opts=opts
        self.cn_models=cn_models
        self.data_sequence=data_sequence


        super(CNV_Evaluator, self).__init__()

    def get_out_data(self):
        parent_exp_dir = './TINYIMAGNET_exp_dir/'  # Change to yours

        dataset_parent_dir = 'Data/TINYIMAGNET'


        dataset_path = os.path.join(dataset_parent_dir, "10", 'trainval_dataset.pth.tar')

        dsets = torch.load(dataset_path)

        self.out_data=dsets

    def evaluate(self,opt):
        all_tasks_results=[]
        #leave last model out of evaluation
        tasks_length=len(self.data_sequence)-1
        for task_index in range(tasks_length):
            in_data_probs=[]
            dataset_path=self.data_sequence[task_index]
            dsets = torch.load(dataset_path)

            cn_model=self.cn_models[task_index]

            cn_model.base_model.to(self.opts.device)
            if task_index == 0:

                nv_model_path = "NoveltyDetectionModules." + opt.novelty_method + "." + opt.novelty_method  # general structure used for package!
                novelty_model = locate(nv_model_path)
                self.get_input_size(cn_model,dsets)
                self.get_out_data()#construct the out dataset using the transformation of the in data.
                self.opts.input_size=self.input_size
                nvmodel = novelty_model(cn_model.base_model,self.opts)
            else:
                nvmodel.model = cn_model.base_model

            nvmodel.train(dsets,self.out_data,task_index)#we should add an out of distribution dataset!
            #evaluate:
            # ---assuming all classes are of the same size
            #creating results module for this current model.
            results=NvDResults(task_index,opt.novelty_method)
            all_tasks_results.append(results)
            #=== hyper  parameters tunning!:
            in_data_prob = nvmodel.get_data_prob(dsets["val"],task_index)#, )
            out_data_prob = nvmodel.get_data_prob(self.out_data["val"],task_index)
            tpr_err = Metrics.tpr95(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end)
            print("out against err",tpr_err)
            results.out_data_tpr=tpr_err
            #==== end of hyper parameters tuning

            #go over all the previous
            for inset_task_index in range(len(cn_model.In)):
                print("inset_task_index",inset_task_index)
                #len(cn_model.In)-len(cn_model.Forgetten)=1
                in_set=cn_model.In[inset_task_index]
                in_data_prob=nvmodel.get_data_prob(in_set,inset_task_index)
                in_data_probs.append(in_data_prob)
                results.det_errs[inset_task_index]=[]
                results.auc[inset_task_index] = []
                results.auprIn[inset_task_index] = []
                results.auprOut[inset_task_index] = []
                results.tpr95[inset_task_index] = []
                #----evaluate in versus out--------
                best_deltas=[]
                for out_set_index in range(len(cn_model.Out)-1):#last task out
                    out_set=cn_model.Outp[out_set_index]
                    print("out_set")

                    out_data_prob = nvmodel.get_data_prob(out_set,inset_task_index)
                    #---evaluate in vs out here-------
                    #det_err=Metrics.tpr95(in_data_prob, out_data_prob, len(dsets['test'].classes),nvmodel.end)
                    det_err,best_delta = Metrics.detection(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end)
                    best_deltas.append(best_delta)#best detlas will be used to the average previous in as out error
                    print("det_err",det_err)
                    #the end needs to be set for ODIN!, only testset has classes, ther others are subsets
                    results.det_errs[inset_task_index].append(det_err)
                    results.tpr95[inset_task_index].append(Metrics.tpr95(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end))
                    results.auprIn[inset_task_index].append(Metrics.auprIn(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end))
                    results.auprOut[inset_task_index].append(Metrics.auprOut(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end))
                    results.auc[inset_task_index].append(Metrics.auroc(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end))
                #---evaluate in versus forgotten
                if inset_task_index<len(cn_model.Forgotten):
                    #a previous task, i.e. there are forgotten samples.
                    print("Forgotten")
                    forgotten_set=cn_model.Forgotten[inset_task_index]
                    forgotten_data_prob = nvmodel.get_data_prob(forgotten_set, inset_task_index)
                    #----false positive rate
                    false_positive_rate,false_negative_rate,best_delta=Metrics.forg_err(in_data_prob, forgotten_data_prob, len(dsets['test'].classes), nvmodel.end)

                    results.false_p_errs.append(false_positive_rate)
                    results.false_n_errs.append(false_negative_rate)

                #specific evaluation to In task
                if inset_task_index==task_index and len(best_deltas)>0:
                    print("current task checking forgotten in")
                    for pre_inset_task_index in range(task_index):
                        #evaluate the current in task again the previous in.
                        print("checking false negative rate",pre_inset_task_index)
                        in_pre_data_prob = in_data_probs[pre_inset_task_index]
                        #---assuming all classes are of the same size
                        #false_negative_rate=1-Metrics.tpr95(in_data_prob, in_pre_data_prob, len(dsets['test'].classes), nvmodel.end)
                        false_negative_rate = Metrics.in_as_out_err( in_pre_data_prob,   best_deltas)

                        results.false_againtcurrent_n_errs.append(false_negative_rate)

                torch.cuda.empty_cache()
            cn_model.base_model.cpu()
            del nvmodel.model
            torch.cuda.empty_cache()
                #cn_model.results.append(results)
        nvmodel.exit_protocols()
        del nvmodel
        return all_tasks_results
    def get_input_size(self,model,dset_in):

        if not hasattr(model.base_model, "input_size"):
            input_size=dset_in["train"][0][0].size() #access the first element of the dataset to fetch the input size
        else:
            input_size=model.module.input_size
        self.input_size=input_size