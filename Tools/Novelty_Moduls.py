class CN_Model(object):
    """A class that represents the model being tested for novelty detection along with the three sets In,Out,Forgotten
       each of these sets is a list of sets where each belongs to a task.
    """
    def __init__(self,model):
        self.In=[] # index in the list represents the task of which the set data belong to
        self.TrainIn=None#the in set for training i.e. those that are correctly predicted
        self.Out=[]
        self.Forgotten=[]
        self.base_model=model
        self.results=[]#each evaluated novelty method will add the obtained results

        super(CN_Model, self).__init__()



