import torch
import torch.nn as nn

""" Class BasicModelParallel represents a "vanilla" model-parallelized version
of the input model. In this case we simply split the original model across GPUs
with not attempt to implement more exotic model-parallel styles like input
pipelining etc. 

    -- model (nn.Module / nn.Sequential): input model
    -- layer_split (list of ints): index of layer splits for the model. Example:
    [16] means to split layers [0, 15] to cuda:0 and layers [16,] to cuda:1 """

class BasicModelParallel(nn.Module):

    def __init__(self, model, layer_split):
        super(BasicModelParallel, self).__init__()

        assert len(layer_split) == torch.cuda.device_count() - 1

        """ iterating through model features currently requires that input model 
        has an nn.Sequential self.features variable that presents the model 
        architecture or is an nn.Sequential """

        if (not isinstance(model, nn.Sequential)):
            model = model.features

        modules = [i[1] for i in list(model._modules.items())]
        layer_split = [0] + layer_split + [len(modules)]

        self.sub_models = []

        for i in range(len(layer_split) - 1):
            self.sub_models.append(nn.Sequential(*modules[layer_split[i]:layer_split[i + 1]]).cuda(i))

    def forward(self, x):
        for i in range(len(self.sub_models)):
            x = self.sub_models[i](x.cuda(i))

        return x.cpu()

    def parameters(self):
        params = []

        for i in self.sub_models:
            params = params + list(i.parameters())

        return params

    def eval(self):
        for i in self.sub_models:
            i.training = False

    def train(self):
        for i in self.sub_models:
            i.training = True
