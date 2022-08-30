import torch
import time
import numpy as np
from  NoveltyDetectionModules.NvDModules import NvDModule


class Energy(NvDModule):

    def __init__(self, model, opts):
        self.CUDA_DEVICE = opts.device
        self.temper = opts.novelty_temperature
        super(Energy, self).__init__(model, opts)

    to_np = lambda x: x.data.cpu().numpy()
    def get_data_prob(self, data_set, task_index):
    #def testData(model, CUDA_DEVICE, dataloader):
        if self.opts.shared_head:
            head_index = -1
        else:
            head_index = task_index
        predicted_probs=[]
        dset_loader = torch.utils.data.DataLoader(data_set, 1,
                                                       shuffle=False)


        self.model.eval()

        self.model.module.eval()
        for j, data in enumerate(dset_loader):
            images, _ = data
            with torch.no_grad():
                inputs = images.cuda(self.CUDA_DEVICE)
                outputs = self.model(inputs)
                del inputs, images,data,j
                if isinstance(outputs, list):
                    #====adaptation to shared head

                    output = outputs[head_index]
                    del outputs
                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                doutput=output.detach()
                del output
                #=================================================================
                predicted_probs.append(-to_np(( self.temper * torch.logsumexp(doutput / self.temper, dim=1))))
                #==================================================================
                #nnOutputs = doutput.cpu()
                #del doutput
                #npOutputs = nnOutputs.numpy()
                #del nnOutputs
                # npOutputs = npOutputs[0]
                # npOutputs = npOutputs - np.max(npOutputs)
                # npOutputs = np.exp(npOutputs) / np.sum(np.exp(npOutputs))
                # predicted_prob=np.max(npOutputs)
                # predicted_probs.append(predicted_prob)

                del npOutputs
                torch.cuda.empty_cache()
        del dset_loader

        torch.cuda.empty_cache()


        return predicted_probs



