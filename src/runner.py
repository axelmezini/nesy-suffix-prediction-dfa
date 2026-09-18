import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required by torch's deterministic cuBLAS ops

from pathlib import Path
import torch
# Force fully deterministic/reproducible runs
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_deterministic_debug_mode('error')
torch.set_printoptions(threshold=float('inf'))

from utils.config import Config
from utils.experiment import Experiment
from common.event_log import Log
from common.declare_model import Model
from common.dfa import SymbolicDFA


def main():
    """
    Entry point: for each dataset, builds a DFA
    from its DECLARE model, then runs an Experiment for every combination of
    noise level and alpha value using that dataset's train/test logs
    """
    config = Config('config.yaml')

    for dataset in config.datasets:
        dataset_path = Path(config.root_path) / 'data' / dataset
        full_log = Log(dataset_path / 'log', 'ordered')
        event_names = full_log.define_event_names()

        model_name = 'test_85-100'
        dfa_folder_name = model_name.replace('test', 'DFA')
        dfa_folder = dataset_path / 'model' / dfa_folder_name
        symbolic_dfa = SymbolicDFA(event_names, dfa_folder)

        # Build the DFA from the DECLARE model's LTL formula only if it hasn't
        # been built before; otherwise reuse the cached DFA files on disk
        if not dfa_folder.exists():
            dfa_folder.mkdir(parents=True, exist_ok=True)
            declare_model = Model(dataset_path / 'model', model_name)
            symbolic_dfa.build_from_formula(declare_model.to_ltl())
        else:
            symbolic_dfa.build_from_file()
        deep_dfa = symbolic_dfa.to_deep_dfa(config.device)

        (Path(config.root_path) / 'results' / dataset).mkdir(parents=True, exist_ok=True)

        for noise in config.noise_levels:
            for alpha in config.alpha_levels:
                train_log = Log(dataset_path / 'log' / model_name, f'train_80_n{noise}')
                test_log = Log(dataset_path / 'log' / model_name, 'test_20')
                train_dataset = train_log.encode(event_names)
                test_dataset = test_log.encode(event_names)

                first_prefix = train_log.get_first_prefix()
                prefixes = [first_prefix, first_prefix + 1, first_prefix + 2]

                experiment = Experiment(config, dataset, prefixes, noise, alpha)
                experiment.run(train_dataset, test_dataset, deep_dfa, event_names + ['end'])


if __name__ == '__main__':
    main()
