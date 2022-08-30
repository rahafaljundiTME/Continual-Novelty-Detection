import torch
import os
from pydoc import locate
from NoveltyDetectionModules import Metrics,Max_Softmax,ODIN,NvDModules
from NoveltyDetectionModules.NvDModules import NvDResults
from torchvision import datasets, transforms
"""
A turn around to gather stats of the different set extracted features without the need to compute the ND performance.
"""
class STAT_Evaluator(object):
    """An abstract class encapsulating a novelty detection method
    """
    def __init__(self,cn_models,data_sequence,opts):
        self.opts=opts
        self.cn_models=cn_models
        self.data_sequence=data_sequence


        super(STAT_Evaluator, self).__init__()

    def get_out_data(self):
        normalize = transforms.Normalize(mean=self.opts.data_mean,
                                         std=self.opts.data_std)


        out_data_train = datasets.CIFAR10(root="Data", download=True, transform=transforms.Compose([
                transforms.Resize(self.input_size[2]),#input size should be passed as a parameter!
                transforms.ToTensor(),
                normalize,
            ]))

        out_data_test = datasets.CIFAR10(root="Data", download=True, train = False, transform=transforms.Compose([
                transforms.Resize(self.input_size[2]),#input size should be passed as a parameter!
                transforms.ToTensor(),
                normalize,
            ]))
        train_dataset, val_dataset = torch.utils.data.random_split(out_data_train, [int(len(out_data_train) * 0.8),
                                                                                      int(len(out_data_train) * 0.2)])
        dsets = {}
        dsets['train'] = train_dataset
        dsets['val'] = val_dataset
        dsets['test'] = out_data_test
        dsets['test'].classes=list(range(10))
        self.out_data=dsets

    def evaluate(self,opt,res):

        all_tasks_results=res
        start_index = len(res)
        for task_index in range(start_index,len(self.data_sequence)):
            #in_data_probs=[]
            all_in_data_probs=[]
            all_out_data_probs = []
            all_forgotten_data_probs = []
            dataset_path=self.data_sequence[task_index]
            dsets = torch.load(dataset_path)

            cn_model=self.cn_models[task_index]

            cn_model.base_model.to(self.opts.device)
            if task_index == start_index:

                nv_model_path = "NoveltyDetectionModules." + opt.novelty_method + "." + opt.novelty_method  # general structure used for package!
                novelty_model = locate(nv_model_path)
                self.get_input_size(cn_model,dsets)
                self.get_out_data()#construct the out dataset using the transformation of the in data.
                self.opts.input_size=self.input_size
                nvmodel = novelty_model(cn_model.base_model,self.opts)
            else:
                nvmodel.model = cn_model.base_model
            train_in=cn_model.TrainIn if hasattr(cn_model,"TrainIn") else None
            nvmodel.train(dsets,self.out_data,task_index,train_in)#we should add an out of distribution dataset!
            #evaluate:
            # ---assuming all classes are of the same size
            #creating results module for this current model.
            results=NvDResults(task_index,opt.novelty_method)
            all_tasks_results.append(results)
            #=== hyper  parameters tunning!:
            nvmodel.name="in_val"
            in_data_prob = nvmodel.get_data_prob(dsets["val"],task_index)#, )
            nvmodel.name = "out_val"
            out_data_prob = nvmodel.get_data_prob(self.out_data["val"],task_index)
            
            #==== end of hyper parameters tuning

            #go over all the previous

            for inset_task_index in range(len(cn_model.In)):
                print("inset_task_index",inset_task_index)
                #len(cn_model.In)-len(cn_model.Forgetten)=1
                in_set=cn_model.In[inset_task_index]
                nvmodel.name="in"+str(task_index)+"_"+str(inset_task_index)
                in_data_prob=nvmodel.get_data_prob(in_set,inset_task_index)
                all_in_data_probs.extend(in_data_prob)
                #in_data_probs.append(in_data_prob)
                results.det_errs[inset_task_index]=[]
                results.auc[inset_task_index] = []
                results.auprIn[inset_task_index] = []
                results.auprOut[inset_task_index] = []
                results.tpr95[inset_task_index] = []


                #----evaluate in versus out--------

                best_deltas=[]
                temp_out=[]
                for out_set in cn_model.Out:
                    print("out_set")
                    nvmodel.name="out"+str(task_index)+"_"+str(inset_task_index)+"_"+str(len(best_deltas)+1)
                    out_data_prob = nvmodel.get_data_prob(out_set,inset_task_index)
                    temp_out.extend(out_data_prob)
                    best_deltas.append(0)
                    #---evaluate in vs out here-------
                    #det_err=Metrics.tpr95(in_data_prob, out_data_prob, len(dsets['test'].classes),nvmodel.end)

                #---evaluate in versus forgotten
                if inset_task_index<len(cn_model.Forgotten):
                    #a previous task, i.e. there are forgotten samples.
                    print("Forgotten")
                    forgotten_set=cn_model.Forgotten[inset_task_index]
                    nvmodel.name = "Forgotten" +str(task_index)+"_"+ str(inset_task_index)
                    forgotten_data_prob = nvmodel.get_data_prob(forgotten_set, inset_task_index)
                    all_forgotten_data_probs.extend(forgotten_data_prob)
                    #----false positive rate
 

  
            cn_model.base_model.cpu()
            del nvmodel.model
            torch.cuda.empty_cache()
                #cn_model.results.append(results)
            torch.save(all_tasks_results,self.output_name)
            torch.save(nvmodel,"NVMODEL_"+self.output_name)# (TEMPOERARYYYYYY)
        nvmodel.exit_protocols()
        del nvmodel
        return all_tasks_results
    def get_input_size(self,model,dset_in):

        if not hasattr(model.base_model, "input_size"):
            input_size=dset_in["train"][0][0].size() #access the first element of the dataset to fetch the input size
        else:
            input_size=model.module.input_size
        self.input_size=input_size


