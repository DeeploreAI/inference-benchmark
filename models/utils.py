# Local lib
from models.common import *


MODULE_MAP = {
    'Linear': Linear,
    'Conv': Conv,
    'Bottleneck': Bottleneck,
    'ResNetLayer': ResNetLayer,
    'AvgPool': AvgPool,
    'MaxPool': MaxPool,
    'Upsample': Upsample,
    'Concat': Concat,
    'Flatten': Flatten,
}


def parse_module(module_name, *args, **kwargs):
    if module_name not in MODULE_MAP:
        raise ValueError(f'Unknown module name: {module_name}')
    else:
        module_class = MODULE_MAP[module_name]
        module = module_class(*args, **kwargs)
        return module
