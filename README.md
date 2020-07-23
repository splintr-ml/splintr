# Splintr - One-Line Distributed Machine Learning and ML Optimization Techniques

Splintr is a Python package for use with PyTorch with the goal of making distributed ML and ML optimization easier, while also giving people access to new and cutting edge techniques in Machine Learning optimization. Currently the package is in its early stages so rapid changes may occur, but we have created a framework that we believe abstracts away complexity very effectively.

## Installation

Splintr is on PyPI and can be installed using pip:
``` pip install splintr ```

## Examples

Currently we only support multi-gpu training and we split our multi-gpu methods between Data Parallel and Model Parallel methods.

### Data Parallel (splintr.DataParallel)

We have three multi-gpu algorithms implemented, which can be selected using the ```style``` argument:

Basic PyTorch Data Parallelism:

```
import splintr

model = # your model here
model = splintr.DataParallel(style = 'torch', model = model, device_ids = None, output_device = None) 

# model behaves identically to normal torch.nn.Module
```

Args: same as PyTorch Data Parallelism (https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed)

Local SGD Data Parallelism (arXiv:1808.07217):

```
import splintr

model = # your model here
model = splintr.DataParallel(style = 'localSGD', model = model, iterations = 40, initializer = lambda x: x) 

# model behaves identically to normal torch.nn.Module
```

Args (these apply Independent Subnet Training as well): model : the input model, iterations : number of local minibatches before universal update (default 40), initializer : weight initialilizer for layers (default ``` lambda x: x```)

Independent Subnet Training For Fully-Connected Layers + Local SGD (arXiv:1910.02120):

```
import splintr

model = # your model here
model = splintr.DataParallel(style = 'ist', model = model, iterations = 40, initializer = lambda x: x) 

# model behaves identically to normal torch.nn.Module
```

### Model Parallel (splintr.ModelParallel)

```
import splintr

model = # your model here
model = splintr.ModelParallel(style = 'basic', model = model, layer_split = [10, 20]

# model behaves identically to normal torch.nn.Module
```

Args: model : the input model, layer_split : the indexes at which to split the layers of the input, in the example above layers[0:10] will be on GPU:0, layers[10:20] will be on GPU:1, and layers[20:] will be on GPU:2.

## Benchmarks (splintr.benchmarks)

We currently have one benchmark dataset (CIFAR 100) and one benchmark model (VGG11), but we are looking to expand our suite of benchmarks, and are open to recommendations.

## Goals/Moving Forward

In terms of future updates to Splintr we have these features slated for the future, and are open to suggestions:
- Pipeline Parallelism, i.e. GPipe, Pipedream (as part of ModelParallel)
- Quantization
- Lottery Ticket Hypothesis
- Pruning
- Memory Optimizations, i.e. Microsoft's ZeRO
- Multi-Machine Parallelism as well as Multi-GPU
- Deep Compression
- More Benchmark Datasets and Models

Created By Varun Srinivasan (personal.varunsrinivasan@gmail.com)
