import torch.nn as nn
import splintr.dataparallel.isubnet as isub
import splintr.dataparallel.local_sgd as lsgd


""" DataParallel returns appropriate DataParallel training option based on user
input. Currently Supports:

	-- 'torch': PyTorch style Data Parallelism
	-- 'localSGD': Vanilla Local SGD (arXiv:1808.07217)
	-- 'ist': Indepent Subnet Training W/ Vanilla Local SGD (arXiv:1910.02120) """

def DataParallel(**kwargs):

	if kwargs['style'] == 'torch':
		model = kwargs['model']
		device_ids = kwargs['device_ids'] if 'device_ids' in kwargs else None
		output_device = kwargs['output_device'] if 'output_device' in kwargs else None
		dim = kwargs['dim'] if 'dim' in kwargs else 0

		return nn.DataParallel(model.cuda(), device_ids = device_ids, output_device = output_device, dim = dim)

	elif kwargs['style'] == 'localSGD':
		model = kwargs['model']
		iterations = kwargs['iterations'] if 'iterations' in kwargs else 40
		initializer = kwargs['initializer'] if 'initializer' in kwargs else lambda x: x

		return lsgd.LocalSGD(model, iterations = iterations, initializer = initializer)

	elif kwargs['style'] == 'ist':
		model = kwargs['model']
		iterations = kwargs['iterations'] if 'iterations' in kwargs else 40
		initializer = kwargs['initializer'] if 'initializer' in kwargs else lambda x: x

		return isub.ISubnet(model, iterations = iterations, initializer = initializer)

	else:
		raise NotImplementedError("No other Data Parallel training methods have been implemented yet!")
