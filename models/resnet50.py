# Basic lib
import yaml

# DL lib
import torch

# Local lib
from models.utils import parse_module
from models.common import *


class ResNet50(nn.Module):
    """Standard ResNet-50 model."""

    def __init__(self, config_path="configs/resnet50.yaml"):
        super().__init__()
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Parse backbone model
        backbone_modules = []
        for i, (n_repeat, module_name, args) in enumerate(config["backbone"]):  # from, number, module, args
            module = parse_module(module_name, *args)
            for _ in range(n_repeat):
                backbone_modules.append(module)
        # self.backbone = nn.Sequential(*backbone_modules)
        self.backbone_modules = nn.ModuleList(backbone_modules)  # TODO: use pytorch hooks to print the hidden layer results, or use torchinfo to print a table of each module

        # Parse task head
        head_modules = []
        for i, (n_repeat, module_name, args) in enumerate(config['head']):
            module = parse_module(module_name, *args)
            for _ in range(n_repeat):
                head_modules.append(module)
        # self.head = nn.Sequential(*head_modules)
        self.head_modules = nn.ModuleList(head_modules)

    def forward(self, x):
        for module in self.backbone_modules:
            x = module(x)
        for module in self.head_modules:
            x = module(x)
        return x