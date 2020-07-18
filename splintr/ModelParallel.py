from splintr.modelparallel.basic_model_parallel import BasicModelParallel

""" ModelParallel returns appropriate ModelParallel training option based on user
input. Currently Supports:

    -- 'basic': Basic Model Parallelism """

def ModelParallel(**kwargs):

    if kwargs['style'] == 'basic':
        model = kwargs['model']
        layer_split = kwargs['layer_split']

        return BasicModelParallel(model, layer_split)

    else:
        raise NotImplementedError("No other Model Parallel training methods have been implemented yet!")
