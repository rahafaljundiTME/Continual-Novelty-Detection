class NvDModule(object):
    """An abstract class encapsulating a novelty detection method
    """
    def __init__(self,model,opts):
        self.opts=opts
        self.model=model
        self.end=1
        self.input_size=opts.input_size
        self.data_mean=opts.data_mean
        self.data_std=opts.data_std
        self.z_size = opts.novelty_z_size

        super(NvDModule, self).__init__()

    def train(self,dset_in,dset_out,task_index,train_in):

        return

    def exit_protocols(self):
        return
    def get_data_prob(self,data_set,task_index):

        return


class NvDResults(object):
    """An abstract class encapsulating a novelty detection method results
    """
    def __init__(self,task_index,method_name):
        self.det_errs={}
        self.auc = {}
        self.auprIn = {}
        self.auprOut = {}
        self.tpr95 = {}
        self.combined_det_err=0#one det err for each model
        self.combined_auc = 0#one auc for each model
        self.combined_false_n_errs=0#one auc for each model
        self.combined_false_p_errs = 0  # one auc for each model
        #===================
        self.false_n_errs=[]
        self.false_againtcurrent_n_errs = []
        self.false_p_errs = []
        self.false_out_n_errs=[]
        self.false_out_p_errs = []
        self.in_forg_auc=[]
        self.out_forg_auc = []
        self.task_index=task_index
        self.method_name=method_name
        self.out_data_tpr=0
        self.out_data_auc=0
        super(NvDResults, self).__init__()
        res=1
        nres=None
        #[print( "model", i, "task", j,"naive",sum(res[i].det_errs[j]) / len(res[i].det_errs[j]),"norm",sum(onres[i].det_errs[j]) / len(onres[i].det_errs[j]),"norm delta",sum(ondres[i].det_errs[j]) / len(ondres[i].det_errs[j])) for i in range(len(res)) for j in range(len(res[i].det_errs.keys()))]
    def report(self):
        print("In vs Out")
        for task in range(0,self.task_index+1):
            if task in self.det_errs.keys():
                print("task",task)
                print("det_errs",self.det_errs[task])
                print("auc", self.auc[task])
                print("auprIn", self.auprIn[task])
                print("auprOut", self.auprOut[task])
                print("tpr95", self.tpr95[task])
            if task<len(self.false_p_errs):
                print("task", task)
                print("rate of forgotten samples confused as in at 95% tpr", self.false_p_errs[task])
            if task < len(self.false_n_errs):
                print("task", task)
                print("rate of remembered samples confused as out at 95% tpr", self.false_n_errs[task])




