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
            tpr_err = Metrics.tpr95(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end)
            auc = Metrics.auroc(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end)
            print("out against err",tpr_err)
            results.out_data_tpr=tpr_err
            results.out_data_auc=auc
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
                all_out_data_probs.append(temp_out)#it will be a list for previous used heads
                #---evaluate in versus forgotten
                if inset_task_index<len(cn_model.Forgotten):
                    #a previous task, i.e. there are forgotten samples.
                    print("Forgotten")
                    forgotten_set=cn_model.Forgotten[inset_task_index]
                    nvmodel.name = "Forgotten" +str(task_index)+"_"+ str(inset_task_index)
                    forgotten_data_prob = nvmodel.get_data_prob(forgotten_set, inset_task_index)
                    all_forgotten_data_probs.extend(forgotten_data_prob)
                    #----false positive rate
                    false_positive_rate,false_negative_rate,best_delta=Metrics.forg_err(in_data_prob, forgotten_data_prob, len(dsets['test'].classes), nvmodel.end)

                    results.false_p_errs.append(false_positive_rate)
                    results.false_n_errs.append(false_negative_rate)
                    if len(cn_model.Out)>0:
                        #---not the last task, there are still out of distribution tasks
                        false_out_positive_rate, false_out_negative_rate, best_delta = Metrics.forg_err(forgotten_data_prob,
                                                                                            all_out_data_probs[inset_task_index],
                                                                                            len(dsets['test'].classes),
                                                                                            nvmodel.end)
                        results.false_out_p_errs.append(false_out_positive_rate)
                        results.false_out_n_errs.append(false_out_negative_rate)
                # #specific evaluation to In task
                # if inset_task_index==task_index and len(best_deltas)>0:
                #     print("current task checking forgotten in")
                #     for pre_inset_task_index in range(task_index):
                #         #evaluate the current in task again the previous in.
                #         print("checking false negative rate",pre_inset_task_index)
                #         in_pre_data_prob = in_data_probs[pre_inset_task_index]
                #         #---assuming all classes are of the same size
                #         #false_negative_rate=1-Metrics.tpr95(in_data_prob, in_pre_data_prob, len(dsets['test'].classes), nvmodel.end)
                #         false_negative_rate = Metrics.in_as_out_err( in_pre_data_prob,   best_deltas)
                #
                #         results.false_againtcurrent_n_errs.append(false_negative_rate)
                #torch.cuda.empty_cache()

            #------------evaluating all pre+current task combined
            #-num_classes all classes so far
            num_classes = nvmodel.model.module.classifier._modules[nvmodel.model.last_layer_index].out_features
            if len(all_out_data_probs) > 0 and  len(cn_model.Out)>0:
                merged_out_probs = []
                x=[merged_out_probs.extend(el) for el in all_out_data_probs]
                #this is done with the all_out_data_probs computed on the each task head in case of multi-head

                det_err, best_delta = Metrics.detection(all_in_data_probs, merged_out_probs,num_classes,
                                                        nvmodel.end)
                results.combined_det_err =det_err
                results.combined_auc= Metrics.auroc(all_in_data_probs, merged_out_probs, num_classes, nvmodel.end)

            if len(all_forgotten_data_probs)>0:

                false_positive_rate, false_negative_rate, best_delta = Metrics.forg_err(all_in_data_probs,
                                                                                        all_forgotten_data_probs,
                                                                                        num_classes,
                                                                                        nvmodel.end)
                results.combined_false_n_errs=false_negative_rate
                results.combined_false_p_errs=false_positive_rate

            cn_model.base_model.cpu()
            del nvmodel.model
            torch.cuda.empty_cache()
                #cn_model.results.append(results)
            torch.save(all_tasks_results,self.output_name)
            torch.save(nvmodel,"NVMODEL_"+self.output_name)
        nvmodel.exit_protocols()
        del nvmodel
        return all_tasks_results
    def get_input_size(self,model,dset_in):

        if not hasattr(model.base_model, "input_size"):
            input_size=dset_in["train"][0][0].size() #access the first element of the dataset to fetch the input size
        else:
            input_size=model.module.input_size
        self.input_size=input_size



class Tuned_CNV_Evaluator(object):
    """An abstract class encapsulating a novelty detection method
    """
    def __init__(self,cn_models,data_sequence,opts):
        self.opts=opts
        self.cn_models=cn_models
        self.data_sequence=data_sequence


        super(Tuned_CNV_Evaluator, self).__init__()

    def get_out_data(self):


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
            all_in_data_probs = []
            all_out_data_probs=[]
            all_forgotten_data_probs=[]
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
            auc=Metrics.auroc(in_data_prob, out_data_prob, len(dsets['test'].classes), nvmodel.end)
            print("out against err",tpr_err)
            results.out_data_tpr=tpr_err
            results.out_data_auc=auc
            #==== end of hyper parameters tuning
            dset_classes=0
            #go over all the previous
            for inset_task_index in range(len(cn_model.In)):
                print("inset_task_index",inset_task_index)
                #len(cn_model.In)-len(cn_model.Forgetten)=1
                in_set=cn_model.In[inset_task_index]
                in_data_prob=nvmodel.get_data_prob(in_set,inset_task_index)
                in_data_probs.append(in_data_prob)
                all_in_data_probs.extend(in_data_prob)
                results.det_errs[inset_task_index]=[]
                results.auc[inset_task_index] = []
                results.auprIn[inset_task_index] = []
                results.auprOut[inset_task_index] = []
                results.tpr95[inset_task_index] = []
                #----evaluate in versus out--------
                best_deltas=[]
                for out_set_index in range(len(cn_model.Out)-1):#last task out
                    out_set=cn_model.Out[out_set_index]
                    print("out_set")

                    out_data_prob = nvmodel.get_data_prob(out_set,inset_task_index)
                    all_out_data_probs.extend(out_data_prob)
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
                    all_forgotten_data_probs.extend(forgotten_data_prob)
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
                #------------evaluating all pre+current task combined
                #-num_classes all classes so far
                num_classes=nvmodel.model.module.classifier._modules[nvmodel.model.last_layer_index].out_features
                det_err, best_delta = Metrics.detection(all_in_data_probs, all_out_data_probs,num_classes,
                                                        nvmodel.end)
                results.combined_det_err =det_err
                results.combined_auc= Metrics.auroc(all_in_data_probs, all_out_data_probs, num_classes, nvmodel.end)
                false_positive_rate, false_negative_rate, best_delta = Metrics.forg_err(all_in_data_probs,
                                                                                        all_forgotten_data_probs,
                                                                                        num_classes,
                                                                                        nvmodel.end)
                results.combined_false_n_errs=false_negative_rate
                results.combined_false_p_errs=false_positive_rate

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