import copy
import torch
import threading
import torch.nn as nn
import splintr.dataparallel.local_sgd as lsgd

""" We implement multi-GPU indepent-subnet training here as tt is described in:
    arXiv:1910.02120. """

""" Class ISubnetFC constitutes the independently trained subcomponents of the
original model's fully-connected components.

    -- in_features (int): number of features in the input Tensor
    -- out_features (int): desired number of features in the output Tensor
    -- back (ISubnetFC): internal variable to map ISubnetFC layers before this
    -- activation (nn.Module / nn.Functional): activation function """

class ISubnetFC(nn.Module):

    def __init__(self, in_features, out_features, back, activation = None):
        super(ISubnetFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # "Master" Weight, utilized for evaluation, etc.

        self.weight = torch.zeros(out_features, in_features)
        nn.init.normal_(self.weight, 0, 0.01)

        # Correct for distributional shift referenced in paper

        self.norm = nn.BatchNorm1d(num_features = out_features)

        self.back = back
        self.final = False

        self.activation = activation if activation else lambda x: x

        """ Sets up initialization of masks, sync's input mask with previous output
        mask """

        self.input_mask = torch.ones(in_features) if not self.back else self.back.mask_sync()
        self.output_mask = torch.randint(low = 0, high = torch.cuda.device_count(), size = (out_features,))

        self.split = torch.cuda.device_count()

        """ mask_map maintains state information about weights and masks across
        devices """

        self.mask_map = dict()
        self.params = [list(self.norm.parameters())]

        for i in range(self.split):
            weight = self.weight.clone().detach().cuda(i)
            weight.requires_grad = True

            if self.back:
                self.mask_map[i] = (weight, (self.input_mask == i).clone().cuda(i), (self.output_mask == i).clone().cuda(i))
            else:
                self.mask_map[i] = (weight, self.input_mask.clone().cuda(i), (self.output_mask == i).clone().cuda(i))

            self.params.append([weight])

    def forward(self, x):
        if self.training:
            x = torch.cuda.comm.scatter(x, devices = range(self.split))
            outputs = [None] * self.split

            def thread(inp, num):
                weight, input_mask, output_mask = self.mask_map[num]
                outputs[num] = (input_mask * inp).matmul(weight.t()) * output_mask

            threads = [threading.Thread(target = thread, args = (x[i], i)) for i in range(self.split)]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            return self.activation(self.norm(torch.cuda.comm.gather(outputs, destination = -1)))
        else:
            return self.activation(self.norm(x.matmul(self.weight.t())))

    def parameters(self):
        params = []

        for i in self.params:
            params = params + i

        return params

    """ "Resets" the masks and "master" weights to propogate updates and start 
    next "cycle" of masking """

    def reset(self):
        index_masks = []

        for i in range(self.split):
            _, input_mask, output_mask = self.mask_map[i]
            output_mask = output_mask.resize(self.out_features, 1)
            index_masks.append(input_mask * output_mask)

        for i in range(self.split):
            self.weight.data[index_masks[i].bool().cpu()] = self.mask_map[i][0].data.clone()[index_masks[i].bool()].cpu()

        self.input_mask = torch.ones(self.in_features) if not self.back else self.back.mask_sync()
        self.output_mask = torch.randint(low = 0, high = torch.cuda.device_count(), size = (self.out_features,)) if not self.final else torch.ones(self.out_features)

        for i in range(self.split):
            self.mask_map[i][0].data = self.weight.data.clone().cuda(i)

            input_mask = (self.input_mask == i).clone().cuda(i) if self.back else self.input_mask.clone().cuda(i)
            output_mask = (self.output_mask == i).clone().cuda(i) if not self.final else self.output_mask.clone().cuda(i)

            self.mask_map[i] = (self.mask_map[i][0], input_mask, output_mask)

    """ Sets up "forward" variable for any "ahead" ISubnetFC layers for mask
    syncing """

    def add_forw(self, forw):
        self.forw = forw

    """ Serves output_mask to sync with self.forw """

    def mask_sync(self):
        return self.output_mask

    """ Designates this ISubnetFC as last in chain so that output_mask is 
    configured correctly """

    def make_final(self):
        self.final = True
        self.output_mask = torch.ones(self.out_features)

        for i in range(self.split):
            self.mask_map[i] = (self.mask_map[i][0], self.mask_map[i][1], self.output_mask.clone().cuda(i))
            

""" Class ISubnet represents the model class that is returned to the user. It
separates the input model appropriately and acts as the interface between the
corresponding ISubnetNFC and ISubnetFC instances. 

    -- model (nn.Module / nn.Sequential): input model for this component
    -- iterations (int): number of iterations before merging weights in local SGD
    and between remasking for the fully-connected layers
    -- initializer (function): Weight initializer. During benchmarking, we found
    that separately initializing the model replicas weight produced better
    results """

class ISubnet(nn.Module):

    def __init__(self, model, iterations = 40, initializer = lambda x: x):
        super(ISubnet, self).__init__()

        self.iterations = iterations
        self.iteration_count = 0
        self.non_subnet_features = []
        self.subnet_features = []

        found_linear = False
        curr_linear = None
        back_linear = None
        fc = None

        """ iterating through model features to find non-subnet features and 
        subnet features currently requires that input model has an nn.Sequential
        self.features variable that presents the model architecture or is an 
        nn.Sequential """

        if (not isinstance(model, nn.Sequential)):
            model = model.features

        for _, layer in model._modules.items():

            if not found_linear:
                if isinstance(layer, nn.Linear):
                    found_linear = True
                    curr_linear = layer

                else:
                    self.non_subnet_features.append(layer)

            else:
                if curr_linear:

                    if isinstance(layer, nn.Linear):
                        fc = ISubnetFC(curr_linear.in_features, curr_linear.out_features, back_linear)
                        curr_linear = layer

                    else:
                        fc = ISubnetFC(curr_linear.in_features, curr_linear.out_features, back_linear, layer)
                        curr_linear = None

                    if back_linear:
                        back_linear.add_forw(fc)

                    self.subnet_features.append(fc)
                    back_linear = fc

                else:
                    if isinstance(layer, nn.Linear):
                        curr_linear = layer

        if curr_linear:
            fc = ISubnetFC(curr_linear.in_features, curr_linear.out_features, back_linear)

            if back_linear:
                back_linear.add_forw(fc)

            self.subnet_features.append(fc)

        if fc:
            fc.make_final()

        self.non_subnet_features = lsgd.LocalSGD(model = nn.Sequential(*self.non_subnet_features), iterations = self.iterations, initializer = initializer)

    def forward(self, x):

        if self.training:
            if self.iteration_count == self.iterations:
                self.reset()
                self.iteration_count = 0

            x = self.non_subnet_features(x)
            for i in self.subnet_features:
                x = i(x)

            self.iteration_count += 1

            return x
        else:
            with torch.no_grad():
                x = self.non_subnet_features(x)
                for i in self.subnet_features:
                    x = i(x)

                return x

    def parameters(self):
        params = self.non_subnet_features.parameters()

        for i in self.subnet_features:
            params = params + i.parameters()

        return params

    """ Manually setting up eval and train functions for the non-subnet and 
    subnet components as well as normalization layers in subnet components """

    def eval(self):
        self.training = False
        self.non_subnet_features.training = False

        for i in self.subnet_features:
            i.training = False
            i.norm.training = False

    def train(self):
        self.training = True
        self.non_subnet_features.training = True

        for i in self.subnet_features:
            i.training = True
            i.norm.training = True

    """ Calls reset on the sub-components, called after specified number of 
    iterations """

    def reset(self):
        for i in self.subnet_features:
            i.reset()
