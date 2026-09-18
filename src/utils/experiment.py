from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import torch
from common.architecture import LSTM, Transformer
from common.training import train
from common.sampling_evaluation import sample
from loss.local_loss import LocalLogicLoss
from loss.global_loss import GlobalLogicLoss
from utils.result import Result
from utils.plotting import plot_metric_bars


class Experiment:
    def __init__(self, config, dataset_name, prefixes, noise, alpha):
        self.config = config
        self.ds_name = dataset_name
        self.noise = noise
        self.alpha = alpha
        self.prefixes = prefixes
        self.results_df = None
        self.experiment_folder = self.create_experiment_folder()

    def run(self, train_ds, test_ds, tensor_dfa, vocabulary):
        results = []

        for architecture in self.config.architectures:
            for run_id in range(0, self.config.nr_runs):
                run_nr = run_id + 1
                run_folder = self.create_run_folder(run_nr)

                for loss in self.config.losses:
                    g = self.set_seed(run_nr)
                    model_results = self.run_model(train_ds, test_ds, tensor_dfa, vocabulary, run_folder, run_nr, loss, architecture, g)

                    if model_results is None:
                        continue
                    results.extend(model_results)

                self.results_df = pd.DataFrame(results)
                self.results_df = self.results_df.round(10)
                output_path = str(self.experiment_folder / 'results.csv')
                self.results_df.to_csv(output_path, index=False)
                self.plot_results()

    def run_model(self, train_ds, test_ds, tensor_dfa, vocabulary, run_folder, run_id, loss, architecture, g):
        nn = self.define_architecture(architecture, vocabulary, train_ds)
        loss_fn = self.define_loss(loss, nn, tensor_dfa)

        if nn is None or loss_fn is None:
            return None

        start_time = time.perf_counter()
        train_acc, test_acc, nr_epochs = train(nn, train_ds, test_ds, self.config, loss, loss_fn)
        training_time = time.perf_counter() - start_time

        model_results = Result(
            architecture, self.ds_name, self.noise, self.alpha, run_id, loss, train_acc, test_acc, nr_epochs, training_time
        )

        self.test(nn, train_ds, test_ds, model_results, g)
        nn.export(run_folder, run_id, loss)
        return model_results.evaluate_predictions(train_ds, test_ds, tensor_dfa)

    def test(self, nn, train_ds, test_ds, model_results, g):
        for prefix in self.prefixes:
            predictions = {
                'train_temperature': sample(nn, train_ds, prefix, self.config.device, self.config.temperature, g=g),
                'test_temperature': sample(nn, test_ds, prefix, self.config.device, self.config.temperature, g=g),
                'train_greedy': sample(nn, train_ds, prefix, self.config.device),
                'test_greedy': sample(nn, test_ds, prefix, self.config.device)
            }
            model_results.add_predictions(prefix, predictions)

    def define_architecture(self, architecture, vocabulary, train_ds):
        if architecture == 'LSTM':
            return LSTM(len(vocabulary), self.config.hidden_dim).to(self.config.device)
        elif architecture == 'transformer':
            return Transformer(len(vocabulary), 128, 8, 2, 256, int(int(train_ds.size(1)) * 2 + 32)).to(self.config.device)
        else:
            print(f'Architecture "{architecture}" is not supported. See config or defaults.')
            return None

    def define_loss(self, loss, architecture, tensor_dfa):
        if loss == 'baseline':
            return torch.nn.CrossEntropyLoss()
        elif loss == 'GLL':
            return GlobalLogicLoss(architecture, tensor_dfa, self.alpha, self.prefixes)
        elif loss == 'LLL':
            return LocalLogicLoss(tensor_dfa, self.alpha)
        else:
            print(f'Loss "{loss}" is not supported. See config or defaults.')
            return None

    def create_experiment_folder(self):
        alpha_string = str(int(round(self.alpha * 100)))
        folder_name = f'{self.config.timestamp}_noise{self.noise}_alpha{alpha_string}'
        experiment_folder = Path(self.config.root_path) / 'results' / self.ds_name / folder_name
        experiment_folder.mkdir(parents=True, exist_ok=True)
        return experiment_folder

    def create_run_folder(self, run_number):
        run_folder = self.experiment_folder / f'run{run_number}'
        run_folder.mkdir(parents=True, exist_ok=True)
        return run_folder

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        return generator

    def plot_results(self):
        for metric in ['similarity_scaled', 'satisfiability']:
            plot_metric_bars(self.results_df, 'test', metric, self.experiment_folder)