from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
import yaml
import torch


# Default parameter values used to fill in anything missing from a loaded config file
DEFAULTS = {
    'root_path': './',
    'datasets': ['BPIC_2013_closed', 'BPIC_2020_travel', 'sepsis'],
    'noise_levels': [10, 20, 30, 40],
    'alpha_levels': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    'architectures': ['LSTM', 'transformer'],
    'losses': ['baseline', 'GLL', 'LLL'],
    'nr_runs': 15 ,
    'hidden_dim': 100 ,
    'temperature': 0.7,
    'batch_size': 64,
    'nr_epochs': 2000 ,
    'min_epochs': 500 ,
    'lr': 0.0005,
    'patience': 35 ,
    'min_delta': 0.00001 ,
    'min_loss': 0.01
}


class Config(SimpleNamespace):
    """
    Loads a YAML config file, fills in missing values with defaults, and
    exposes all settings as attributes instead of dict keys
    """
    def __init__(self, filepath):
        with open(filepath) as file:
            config = yaml.safe_load(file)

        val_config = self.validate(config)
        super().__init__(**dict_to_ns(val_config).__dict__)

    def save(self, filepath):
        """
        Writes the current config back out to a YAML file
        """
        with open(filepath, 'w') as file:
            yaml.safe_dump(ns_to_dict(self), file, sort_keys=False)

    def validate(self, config):
        """
        Fills gaps in the loaded config: adds a run timestamp, picks the compute
        device, and applies DEFAULTS for any key not already present in the file
        """
        v_cfg = deepcopy(config) if config else {}

        v_cfg['timestamp'] = datetime.now().strftime('%m%d-%H%M%S')
        v_cfg['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for k, default in DEFAULTS.items():
            if k not in v_cfg:
                print(f'Value for parameter {k} not found in config file. Set default to: {default}.')
                v_cfg[k] = deepcopy(default)
        return v_cfg

def dict_to_ns(d):
    """
    Recursively converts a (possibly nested) dict into nested SimpleNamespace objects,
    so config values can be accessed as attributes
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d

def ns_to_dict(obj):
    """
    Inverse of dict_to_ns: recursively converts nested SimpleNamespace/dict/list
    structures back into plain dicts/lists so they can be YAML-serialized;
    anything not a basic type is stringified as a fallback
    """
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: ns_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ns_to_dict(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
