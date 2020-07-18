import copy
import torch
import threading
import torch.nn as nn

""" Class LocalSGD implements Vanilla Local SGD as described in 
(arXiv:1808.07217) for multi-GPU training of sequential models.

    -- model (nn.Sequential): input model for this component
    -- iterations (int): local SGD iterations between merging updates
    -- initializer (function): Weight initializer. During benchmarking, we found
    that separately initializing the model replicas weight produced better
    results """   

class LocalSGD(nn.Module):

    def __init__(self, model, iterations = 40, initializer = lambda x: x):
        super(LocalSGD, self).__init__()
        self.split = torch.cuda.device_count()
        self.broadcast_map = dict()
        self.iterations = iterations
        self.iteration_count = 0

        """ broadcast_map maintains state of model instances across devices """

        for i in range(self.split):
            self.broadcast_map[i] = copy.deepcopy(model).cuda(i)
            initializer(self.broadcast_map[i])

        self.params = []

        for i in range(self.split):
            self.params.append(list(self.broadcast_map[i].parameters()))

    def forward(self, x):

        if self.training:
            if self.iteration_count == self.iterations:
                self.reset()
                self.iteration_count = 0

            x = torch.cuda.comm.scatter(x, devices = list(range(self.split)))
            outputs = [None] * self.split

            def thread(inp, num):
                outputs[num] = self.broadcast_map[num](inp)

            threads = [threading.Thread(target = thread, args = (x[i], i)) for i in range(self.split)]

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            self.iteration_count += 1
                
            return torch.cuda.comm.gather(outputs, destination = -1)
        else:
            x = x.cuda(0)
            return self.broadcast_map[0](x).cpu()

    def parameters(self):
        params = []

        for i in self.params:
            params = params + i

        return params

    """ Reset performs a local-SGD iteration across model instances """

    def reset(self):
        param_data = None

        for i in range(len(self.params[0])):
            for j in range(self.split):
                if param_data != None:
                    param_data.data = param_data.data + self.params[j][i].data.clone().cpu()
                else:
                    param_data = self.params[j][i].data.clone().cpu()
                    
            param_data.data = param_data.data / self.split
            
            for j in range(self.split):
                self.params[j][i].data = param_data.data.clone().cuda(j)
                
            param_data = None