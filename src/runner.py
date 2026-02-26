import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_deterministic_debug_mode("error")
torch.set_printoptions(threshold=float('inf'))

from utils.config import Config
from utils.experiment import Experiment
from common.event_log import Log
from common.declare_model import Model
from common.dfa import SymbolicDFA


def main():
    config = Config('config.yaml')

    for dataset in config.datasets:
        full_log = Log(config.root_path, dataset, 'ordered')
        event_names = full_log.define_event_names()

        dfa_folder_name = f'DFA_{config.template_type}_{config.template_support}/'
        dfa_folder = os.path.join(config.root_path, 'datasets', dataset, 'model', dfa_folder_name)
        symbolic_dfa = SymbolicDFA(event_names, dfa_folder)

        if not os.path.exists(dfa_folder):
            os.makedirs(dfa_folder)
            declare_model = Model(config.root_path, dataset, config.template_type, config.template_support)
            symbolic_dfa.build_from_formula(declare_model.to_ltl())
        else:
            symbolic_dfa.build_from_file()
        deep_dfa = symbolic_dfa.to_deep_dfa(config.device)

        os.makedirs(os.path.join(config.root_path, 'results', dataset), exist_ok=True)

        for noise in config.noise_levels:
            for alpha in config.alpha_levels:
                train_log = Log(config.root_path, dataset, f'train_80_all_{noise}n')
                test_log = Log(config.root_path, dataset, f'test_20_all')
                train_dataset = train_log.encode(event_names)
                test_dataset = test_log.encode(event_names)

                first_prefix = train_log.get_first_prefix()
                prefixes = [first_prefix, first_prefix + 1, first_prefix + 2]

                experiment = Experiment(config, dataset, prefixes, noise, alpha)
                experiment.run(train_dataset, test_dataset, deep_dfa, event_names + ['end'])


def print_start_acceptance(train_dataset, test_dataset, deep_dfa, config):
    for ds_name, ds in [("train", train_dataset), ("test", test_dataset)]:
        __, dfa_rew = deep_dfa.forward_pi(ds.to(config.device))
        prob_acceptance = dfa_rew[:, -1, 1]
        loss = -torch.log(prob_acceptance.clamp(min=1e-10)).mean()
        print(f"{ds_name.capitalize()} Set Mean Acceptance: {prob_acceptance.mean().item():.6f}, Ground Loss: {loss.item():.6f}\n")


if __name__ == '__main__':
    main()
