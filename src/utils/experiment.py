import os
import time
import random
import numpy as np
import pandas as pd
import torch
from common.architecture import LSTM, Transformer
from common.training import train_new
from common.sampling import sample
from utils.result import Result
from utils.plotting import plot_metrics_as_bars


#TODO: train_ds, test_ds, dfa in self? da vedere dopo aver aggiunto il noise; salavare tracce predette e plot loss
class Experiment:
    def __init__(self, config, dataset_name, prefixes, noise, alpha):
        self.config = config
        self.ds_name = dataset_name
        self.noise = noise
        self.alpha = alpha
        self.prefixes = prefixes
        self.results_df = None
        self.experiment_folder = self.create_experiment_folder()
        self._stop_event = None

    def run(self, train_ds, test_ds, tensor_dfa, vocabulary):
        self._stop_event = [0] * (len(vocabulary)-1) + [1]
        results = []

        for run_id in range(0, self.config.nr_runs):
            run_nr = run_id + 1
            run_folder = self.create_run_folder(run_nr)

            for mode in self.config.modes:
                g = self.set_seed(run_nr)
                if self.config.architecture == 'LSTM':
                    model = LSTM(len(vocabulary), self.config.hidden_dim).to(self.config.device)
                else:
                    model = Transformer(len(vocabulary), 128, 8, 2, 256, int(int(train_ds.size(1)) * 2 + 32)).to(self.config.device)
                mode_results = self.run_mode(model, train_ds, test_ds, tensor_dfa, run_folder, run_nr, mode, g)
                results.extend(mode_results)

            self.results_df = pd.DataFrame(results)
            self.results_df['dataset'] = self.ds_name
            output_path = os.path.join(self.experiment_folder, 'results.csv')
            self.results_df.to_csv(output_path, index=False)

    def run_mode(self, model, train_ds, test_ds, tensor_dfa, run_folder, run_id, mode, g):
        #model = deepcopy(rnn).to(self.config['device'])
        #TODO: inizializzare qua le varie loss

        start_time = time.perf_counter()
        if mode == 'baseline': #TODO: da generalizzare
            print('baseline started')
            train_acc, test_acc, _, _, nr_epochs = train_new(model, train_ds, test_ds, self.config, mode)
        #elif mode == 'gll_old':
        #    train_acc, test_acc, _, _, _, nr_epochs = train_old(model, train_ds, test_ds, self.config.nr_epochs, self.alpha, deepdfa=tensor_dfa, prefixes=self.prefixes)
        else:
            print(f'{mode} started')
            train_acc, test_acc, _, _, nr_epochs = train_new(model, train_ds, test_ds, self.config, mode, tensor_dfa, self.alpha, self.prefixes)
        training_time = time.perf_counter() - start_time

        mode_result = Result(run_folder, run_id, mode, train_acc, test_acc, nr_epochs, training_time)

        self.test(model, train_ds, test_ds, tensor_dfa, mode_result, mode, g)
        export_model(model, run_folder, f'rnn_{mode}')
        return mode_result.evaluate_predictions(train_ds, test_ds, tensor_dfa)

    def test(self, model, train_ds, test_ds, tensor_dfa, mode_result, mode, g):
        for prefix in self.prefixes:
            if mode == 'LLL' or mode == 'GLL' or mode=='baseline':
                predictions =  {
                    'train_temperature': sample(model, train_ds, prefix, self.config.device, self.config.temperature, g=g),
                    'test_temperature': sample(model, test_ds, prefix, self.config.device, self.config.temperature, g=g),
                    'train_greedy': sample(model, train_ds, prefix, self.config.device, 0),
                    'test_greedy': sample(model, test_ds, prefix, self.config.device, 0)
                }
            else:
                predictions = {
                    #'train_temperature': suffix_prediction_with_temperature_with_stop(model, train_ds, prefix, temperature=self.config.temperature, stop_event=self._stop_event, g=g),
                    #'test_temperature': suffix_prediction_with_temperature_with_stop(model, test_ds, prefix, temperature=self.config.temperature, stop_event=self._stop_event, g=g),
                    #'train_greedy': greedy_suffix_prediction_with_stop(model, train_ds, prefix, stop_event=self._stop_event),
                    #'test_greedy': greedy_suffix_prediction_with_stop(model, test_ds, prefix, stop_event=self._stop_event)
                }
            mode_result.add_predictions(prefix, predictions)

    def save_results(self):
        output_path = os.path.join(self.experiment_folder, 'results.csv')
        self.results_df.to_csv(output_path, index=False)

    def plot_results(self):
        for metric in ['accuracy', 'DL', 'DL scaled', 'sat']:
            filename = os.path.join(self.experiment_folder, f'{metric}_mean.png')
            plot_metrics_as_bars(self.results_df, metric, self.prefixes, filename, errorbar=True)

    def create_experiment_folder(self):
        folder_name = f'{self.config.timestamp}_{self.config.template_type}_n{self.noise}_a{int(round(self.alpha * 100)):02d}'
        experiment_folder = str(os.path.join(self.config.root_path, 'results', self.ds_name, folder_name))
        os.makedirs(experiment_folder)
        return experiment_folder

    def create_run_folder(self, run_number):
        run_folder = os.path.join(self.experiment_folder, f'run{run_number}')
        for subfolder in ['plots', 'predicted_traces', 'models']:
            os.makedirs(os.path.join(run_folder, subfolder))
        return run_folder

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        return generator

def export_model(model, run_folder, name):
    torch.save(model.state_dict(), os.path.join(run_folder, 'models', f'{name}.pt'))
    del model
    torch.cuda.empty_cache()




#def calculate_results(self, record, run_folder):
#    with open(os.path.join(run_folder, 'predicted_traces', f'predicted_{model_type}_{strategy}_{prefix}.txt'), mode='w') as file:
#        file.write(self.log.decode(predicted_test))
