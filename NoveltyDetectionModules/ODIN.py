import torch
import time
import numpy as np
import torch.nn as nn
from  NoveltyDetectionModules.NvDModules import NvDModule

class ODIN(NvDModule):
    """
    """
    def __init__(self,model,opts):
        self.CUDA_DEVICE=opts.device
        self.criterion=nn.CrossEntropyLoss() if not hasattr(opts, "novelty_criterion") else opts.criterion
        self.noiseMagnitude=opts.novelty_magnitude
        self.temper=opts.novelty_temperature
        self.end = 0.12#double number of classes in cifar
        super(ODIN, self).__init__(model,opts)

    def get_data_prob(self, data_set, task_index):
            #testData(model, criterion, CUDA_DEVICE, testloader, noiseMagnitude, temper):
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
            del data,j
            self.model.zero_grad()
            inputs = images.cuda(self.CUDA_DEVICE)
            del images
            inputs.requires_grad_(True)
            outputs = self.model(inputs)
            if isinstance(outputs, list):
                output = outputs[head_index]
                del outputs
            # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
            nnOutputs = output.data.cpu()
            nnOutputs = nnOutputs.numpy()
            nnOutputs = nnOutputs[0]
            nnOutputs = nnOutputs - np.max(nnOutputs)
            nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

            # Using temperature scaling
            output = output / self.temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            maxIndexTemp = np.argmax(nnOutputs)
            labels = torch.LongTensor([maxIndexTemp]).cuda(self.CUDA_DEVICE)
            loss = self.criterion(output, labels)
            loss.backward()
            del labels
            del loss
            del output
            del nnOutputs
            with torch.no_grad():
                # Normalizing the gradient to binary in {0, 1}
                gradient = torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                # Normalizing the gradient to the same space of image
                gradient[0][0] = (gradient[0][0]) / (self.data_std[0])
                gradient[0][1] = (gradient[0][1]) / (self.data_std[1])
                gradient[0][2] = (gradient[0][2]) / (self.data_std[2])
                # Adding small perturbations to images
                tempInputs = torch.add(inputs.data, gradient,alpha= -self.noiseMagnitude)
                del inputs
                del gradient
                outputs = self.model((tempInputs))
                del tempInputs
                if isinstance(outputs, list):
                    output = outputs[head_index]
                output = output / self.temper
                # Calculating the confidence after adding perturbations
                nnOutputs = output.data.cpu()

                del outputs,output
                nnOutputs = nnOutputs.numpy()
                nnOutputs = nnOutputs[0]
                nnOutputs = nnOutputs - np.max(nnOutputs)
                nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
                predicted_prob=np.max(nnOutputs)
                predicted_probs.append(predicted_prob)
                del nnOutputs
            self.model.zero_grad()

        del dset_loader

        torch.cuda.empty_cache()
        return predicted_probs

