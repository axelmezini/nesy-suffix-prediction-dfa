import os
import random
from copy import deepcopy
import numpy as np
import torch
from modules import LSTM
from training import train
from sampling import sample, evaluate_similarity, evaluate_satisfiability

HIDDEN_DIMENSIONS = 100


class Run:
    def __init__(self, experiment_folder, train_dataset, test_dataset, dfa, log, run_number, device):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tensor_dfa = dfa
        self.log = log
        self.run_number = run_number
        self.seed = self.set_seed(run_number)
        self.run_folder = self.create_run_folder(experiment_folder)
        self.device = device

    def run(self, vocabulary, prefixes):
        rnn = LSTM(len(vocabulary), HIDDEN_DIMENSIONS)
        results = []

        # Baseline
        model = deepcopy(rnn).to(self.device)
        train_accuracy, test_accuracy, train_losses, test_losses, nr_epochs = train(model, None, self.train_dataset, self.test_dataset, self.device)
        results.extend(self.test(model, prefixes, 'baseline', train_accuracy, test_accuracy, nr_epochs))

        torch.save(model.state_dict(), os.path.join(self.run_folder, 'models', 'rnn_Baseline.pt'))
        del model
        torch.cuda.empty_cache()

        # With BK
        model = deepcopy(rnn).to(self.device)
        train_accuracy, test_accuracy, train_losses, test_losses, nr_epochs = train(model, self.tensor_dfa, self.train_dataset, self.test_dataset, self.device)
        results.extend(self.test(model, prefixes, 'with BK', train_accuracy, test_accuracy, nr_epochs))

        torch.save(model.state_dict(), os.path.join(self.run_folder, 'models', 'rnn_BK.pt'))
        del model
        torch.cuda.empty_cache()

        return results

    def test(self, model, prefixes, model_type, train_accuracy, test_accuracy, nr_epochs):
        results = []
        for prefix in prefixes:
            train_pred_temp, test_pred_temp = self.run_sampling(model, prefix, 0.7)
            results.append(self.calculate_results(train_pred_temp, test_pred_temp, prefix, model_type, 'temperature', train_accuracy, test_accuracy, nr_epochs))

            train_pred_greedy, test_pred_greedy = self.run_sampling(model, prefix, 0)
            results.append(self.calculate_results(train_pred_greedy, test_pred_greedy, prefix, model_type, 'greedy', train_accuracy, test_accuracy, nr_epochs))
        return results

    def run_sampling(self, model, prefix, temperature):
        train_predictions = sample(model, self.train_dataset, prefix, self.device, temperature, self.tensor_dfa)
        test_predictions = sample(model, self.test_dataset, prefix, self.device, temperature, self.tensor_dfa)
        return train_predictions, test_predictions

    def calculate_results(self, predicted_train, predicted_test, prefix, model, strategy, train_accuracy, test_accuracy, nr_epochs):
        train_dl, train_dl_scaled = evaluate_similarity(predicted_train, self.train_dataset)
        test_dl, test_dl_scaled = evaluate_similarity(predicted_test, self.test_dataset)

        with open(os.path.join(self.run_folder, 'predicted_traces', f'predicted_{model}_{strategy}_{prefix}.txt'), mode='w') as file:
            file.write(self.log.decode(predicted_test))

        return {
            'run_id': self.run_number,
            'prefix length': prefix,
            'model': model,
            'sampling strategy': strategy,
            'train accuracy': train_accuracy,
            'test accuracy': test_accuracy,
            'train DL': train_dl,
            'train DL scaled': train_dl_scaled,
            'test DL': test_dl,
            'test DL scaled': test_dl_scaled,
            'train sat': evaluate_satisfiability(self.tensor_dfa, predicted_train),
            'test sat': evaluate_satisfiability(self.tensor_dfa, predicted_test),
            'nr_epochs': nr_epochs
        }

    def set_seed(self, seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # set to false for reproducibility, True to boost performance
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        random_state = random.getstate()
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        return random_state

    def create_run_folder(self, experiment_folder):
        run_folder = os.path.join(experiment_folder, f'run{self.run_number}')
        os.makedirs(os.path.join(run_folder, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(run_folder, 'predicted_traces'), exist_ok=True)
        os.makedirs(os.path.join(run_folder, 'models'), exist_ok=True)
        return run_folder
