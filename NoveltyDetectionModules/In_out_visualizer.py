import torch
import matplotlib.pyplot as plt#from openTSNE import TSNE
from sklearn.manifold import TSNE
import numpy as np
def get_features(model, data_set,device):


    dset_loader = torch.utils.data.DataLoader(data_set, 1,
                                              shuffle=False)

    model.eval()

    all_features=None
    for j, data in enumerate(dset_loader):
        images, _ = data
        with torch.no_grad():
            inputs = images.to(device)
            features=model.module.return_hidden(inputs)
            if all_features is None:
                all_features=features.clone()
            else:
                all_features =torch.cat( (all_features,features),dim=0)

    return all_features
def visulaize_cnmodel(cn_model,name,device):

    sets = {}
    for inset_task_index in range(len(cn_model.In)):
        sets={}
        print("inset_task_index", inset_task_index)
        # len(cn_model.In)-len(cn_model.Forgetten)=1
        in_set = cn_model.In[inset_task_index]
        sets["in"] = in_set
        for out_set_index in range(len(cn_model.Out) - 1):  # last task out
            out_set = cn_model.Out[out_set_index]
            sets["out"] = out_set
            visualize_embedding(cn_model.base_model,sets,name+"in_"+str(inset_task_index)+"_out_"+str(out_set_index),device)  
        sets={}
        sets["in"]=in_set
        if inset_task_index < len(cn_model.Forgotten):
            # a previous task, i.e. there are forgotten samples.
            print("Forgotten")
            forgotten_set = cn_model.Forgotten[inset_task_index]
            sets["forgotten"] = forgotten_set
        
            visualize_embedding(cn_model.base_model,sets,name+"in_"+str(inset_task_index)+"_forgotten",device)

def visualize_embedding(model,sets,name,device):

    set_features=None
    set_labels=None
    key_indx = 0
    for key in sets.keys():
        this_set_features = get_features(model, sets[key],device).cpu()

        if set_features is None:
            set_features=this_set_features.clone()
            set_labels=torch.zeros(set_features.size(0))
        else:
            set_features = torch.cat((set_features, this_set_features.clone()), dim=0)
            set_labels = torch.cat((set_labels, torch.ones(set_features.size(0))*key_indx), dim=0)
        key_indx+=1


    embedding= TSNE(n_components=2).fit_transform(set_features.numpy())
    #import pdb;pdb.set_trace()
    # Create plot
    fig = plt.figure()

    colors=["C"+str(cl) for cl in range(len(sets.keys()))]
    for data in zip(embedding, set_labels):
        x, y = data
        y=int(y.cpu().numpy())
        plt.scatter(x[0],x[1], alpha=0.8, c=colors[y], edgecolors='none', s=30, label=str(y))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())




    fig.savefig(name)

