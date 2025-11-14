import os
from datetime import datetime
import pandas as pd
import torch
from log import Log
from model import Model
from dfa import SymbolicDFA, TensorDFA
from run import Run
from plotting import plot_metrics_as_bars


def main():
    #root_path = '../'  # local
    root_path = '/data/users/amezini/'  # cluster
    template_type = 'all'
    template_support = '85-100'
    timestamp = datetime.now().strftime('%m%d-%H%M%S')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for dataset in ['sepsis', 'BPIC_2013_closed', 'BPIC_2020_travel']:
        full_log = Log(root_path, dataset, 'ordered')
        event_names = full_log.get_event_names()
        print(len(event_names))

        dfa_folder = os.path.join(root_path, 'datasets', dataset, 'model', f'DFA_{template_type}_{template_support}/')
        if True: #not os.path.isdir(dfa_folder):
            declare_model = Model(root_path=root_path, dataset=dataset, template_type=template_type, template_support=template_support)
            declare_model.to_ltl()
            os.makedirs(dfa_folder, exist_ok=True)
            declare_model.to_dfa(dfa_folder)

        dfa = prepare_dfa(dfa_folder, event_names, device)

        for noise in ['_04n']:

            for alpha in [0.75]:
                config = {
                    'root_path': root_path,
                    'template_type': template_type,
                    'timestamp': timestamp,
                    'device': device,
                    'nr_runs': 1,
                    'dataset': dataset,
                    'log': full_log,
                    'event_names': event_names,
                    'dfa': dfa,
                    'noise': noise,
                    'alpha': alpha,
                }
                run_experiment(config)


def run_experiment(config):
    experiment_folder = create_experiment_folder(config)
    log_train = prepare_log(config, 'train_80', config['noise'])
    log_test = prepare_log(config, 'test_20', '')
    train_dataset = log_train.to_tensor(1)
    test_dataset = log_test.to_tensor(1)

    first_prefix = log_train.get_first_prefix()
    prefixes = [first_prefix, first_prefix + 1, first_prefix + 2]

    data = []
    for run_id in range(config['nr_runs']):
        print(f'Running run {run_id + 1}')
        run = Run(experiment_folder, train_dataset, test_dataset, config['dfa'], config['log'], run_id + 1, config['device'])
        data.extend(run.run(config['event_names'] + ['end'], prefixes, config['alpha']))

    df_results = pd.DataFrame(data)
    df_results['dataset'] = config['dataset']
    save_results(df_results, experiment_folder)
    plot_results(df_results, prefixes, experiment_folder)


def prepare_log(config, portion, noise):
    log_name = f'{portion}_{config["template_type"]}{noise}'
    log = Log(config['root_path'], config['dataset'], log_name)
    log.encode(config['event_names'])
    return log


def prepare_dfa(dfa_path, event_names, device):
    symbolic_dfa = SymbolicDFA(dfa_path, event_names)
    return TensorDFA(symbolic_dfa, device)


def create_experiment_folder(config):
    results_folder = os.path.join(config['root_path'], 'results', config['dataset'])
    os.makedirs(results_folder, exist_ok=True)

    folder_name = f'{config["timestamp"]}_{config["template_type"]}{config["noise"]}_{config["alpha"]*100}_{config["beta"]*100}'
    experiment_folder = os.path.join(str(results_folder), folder_name)

    os.makedirs(experiment_folder)
    return experiment_folder


def save_results(df, experiment_folder):
    output_path = os.path.join(experiment_folder, 'results.csv')
    df.to_csv(output_path, index=False)


def plot_results(df, prefixes, experiment_folder):
    plot_metrics_as_bars(df, 'accuracy', prefixes, os.path.join(experiment_folder, 'accuracy_mean.png'), errorbar=True)
    plot_metrics_as_bars(df, 'DL', prefixes, os.path.join(experiment_folder, 'DL_mean.png'), errorbar=True)
    plot_metrics_as_bars(df, 'DL scaled', prefixes, os.path.join(experiment_folder, 'DL_scaled_mean.png'), errorbar=True)
    plot_metrics_as_bars(df, 'sat', prefixes, os.path.join(experiment_folder, 'sat_mean.png'), errorbar=True)


if __name__ == '__main__':
    main()