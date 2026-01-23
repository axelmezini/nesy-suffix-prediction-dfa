from datetime import datetime
from types import SimpleNamespace
import yaml
import torch


class Config(SimpleNamespace):
    def __init__(self, filepath):
        with open(filepath) as file:
            config = yaml.safe_load(file)

        config['timestamp'] = datetime.now().strftime('%m%d-%H%M%S')
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        super().__init__(**dict_to_ns(config).__dict__)

    def save(self, filepath):
        with open(filepath, 'w') as file:
            yaml.safe_dump(ns_to_dict(self), file, sort_keys=False)

def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d

def ns_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: ns_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ns_to_dict(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)