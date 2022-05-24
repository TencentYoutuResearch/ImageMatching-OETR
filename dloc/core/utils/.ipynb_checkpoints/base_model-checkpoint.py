import inspect
from abc import ABCMeta, abstractmethod
from copy import copy

from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_data_keys = []

    def __init__(self, conf, model_path, device='cuda'):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_data_keys = copy(self.required_data_keys)
        self._init(conf, model_path, device)
        self.model_path = model_path
        self.device = device

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf, model_path):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError


def dynamic_load(root, model):
    module_path = f'{root.__name__}.{model}'
    module = __import__(module_path, fromlist=[''])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)]
    assert len(classes) == 1, classes
    return classes[0][1]
    # return getattr(module, 'Model')
