import os
import time
import random
import numpy as np
import pandas as pd
import torch
from common.architecture import LSTM, Transformer
from common.training import train
from common.sampling import sample
from loss.local_loss import LocalLogicLoss
from loss.global_loss import GlobalLogicLoss
from utils.result import Result
from utils.plotting import plot_metrics_as_bars


class Experiment:
    def __init__(self, config, dataset_name, prefixes, noise, alpha):
        self.config = config
        self.ds_name = dataset_name
        self.noise = int(noise) * 10
        self.alpha = alpha
        self.prefixes = prefixes
        self.results_df = None
        self.experiment_folder = self.create_experiment_folder()

    def run(self, train_ds, test_ds, tensor_dfa, vocabulary):
        results = []

        for run_id in range(0, self.config.nr_runs):
            run_nr = run_id + 1
            run_folder = self.create_run_folder(run_nr)

            for model in self.config.models:
                g = self.set_seed(run_nr)
                model_results = self.run_model(train_ds, test_ds, tensor_dfa, vocabulary, run_folder, run_nr, model, g)
                results.extend(model_results)

            self.results_df = pd.DataFrame(results)
            self.results_df = self.results_df.round(10)
            output_path = os.path.join(self.experiment_folder, 'results.csv')
            self.results_df.to_csv(output_path, index=False)

    def run_model(self, train_ds, test_ds, tensor_dfa, vocabulary, run_folder, run_id, model, g):
        architecture = self.define_architecture(self.config.architecture, vocabulary, train_ds)
        loss_fn = self.define_loss(model, architecture, tensor_dfa)

        start_time = time.perf_counter()
        train_acc, test_acc, nr_epochs = train(architecture, train_ds, test_ds, self.config, model, loss_fn, self.alpha)
        training_time = time.perf_counter() - start_time

        model_results = Result(
            self.config.architecture, self.ds_name, self.noise, self.alpha, run_id, model, train_acc, test_acc, nr_epochs, training_time
        )

        self.test(architecture, train_ds, test_ds, model_results, g)
        architecture.export(self.experiment_folder, run_id, model)
        return model_results.evaluate_predictions(train_ds, test_ds, tensor_dfa)

    def test(self, architecture, train_ds, test_ds, model_results, g):
        for prefix in self.prefixes:
            predictions = {
                'train_temperature': sample(architecture, train_ds, prefix, self.config.device, self.config.temperature, g=g),
                'test_temperature': sample(architecture, test_ds, prefix, self.config.device, self.config.temperature, g=g),
                'train_greedy': sample(architecture, train_ds, prefix, self.config.device, 0),
                'test_greedy': sample(architecture, test_ds, prefix, self.config.device, 0)
            }
            model_results.add_predictions(prefix, predictions)

    def define_architecture(self, architecture_name, vocabulary, train_ds):
        if architecture_name == 'LSTM':
            return LSTM(len(vocabulary), self.config.hidden_dim).to(self.config.device)
        else:
            return Transformer(len(vocabulary), 128, 8, 2, 256, int(int(train_ds.size(1)) * 2 + 32)).to(self.config.device)

    def define_loss(self, model_name, architecture, tensor_dfa):
        if model_name == 'baseline':
            return torch.nn.CrossEntropyLoss()
        elif model_name == 'LLL':
            return LocalLogicLoss(tensor_dfa, self.alpha, self.config.device)
        else:
            return GlobalLogicLoss(architecture, tensor_dfa, self.alpha, self.config.device, self.prefixes)

    def create_experiment_folder(self):
        alpha_string = str(int(round(self.alpha * 100)))
        folder_name = f'{self.config.timestamp}_noise{self.noise}_alpha{alpha_string}'
        experiment_folder = str(os.path.join(self.config.root_path, 'results', self.ds_name, folder_name))
        os.makedirs(experiment_folder)
        return experiment_folder

    def create_run_folder(self, run_number):
        run_folder = os.path.join(self.experiment_folder, f'run{run_number}')
        for subfolder in []: # ['plots', 'predicted_traces', 'models']:
            os.makedirs(os.path.join(run_folder, subfolder))
        return run_folder

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        return generator

    def plot_results(self):
        for metric in ['accuracy', 'DL', 'DL scaled', 'sat']:
            filename = os.path.join(self.experiment_folder, f'{metric}_mean.png')
            plot_metrics_as_bars(self.results_df, metric, self.prefixes, filename, errorbar=True)