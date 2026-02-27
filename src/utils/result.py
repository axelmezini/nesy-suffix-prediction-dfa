from common.sampling import evaluate_similarity, evaluate_satisfiability


class Result:
    def __init__(self, architecture, dataset, noise, alpha, run_id, model, train_acc, test_acc, nr_epochs, training_time):
        self.architecture = architecture
        self.dataset = dataset
        self.noise = noise
        self.alpha = alpha
        self.run_id = run_id
        self.model = model
        self.train_accuracy = train_acc
        self.test_accuracy = test_acc
        self.nr_epochs = nr_epochs
        self.training_time = training_time
        self.predictions = {}

    def add_predictions(self, prefix, predictions):
        self.predictions[prefix] = predictions

    def evaluate_predictions(self, train_ds, test_ds, tensor_dfa):
        results = []
        for prefix, lists in self.predictions.items():
            for strategy in ['temperature', 'greedy']:
                results.append(
                    self.calculate_metrics(prefix, strategy, lists[f'train_{strategy}'], lists[f'test_{strategy}'],
                                           train_ds, test_ds, tensor_dfa)
                )
        return results

    def calculate_metrics(self, prefix, strategy, train_pred, test_pred, train_ds, test_ds, tensor_dfa):
        train_dl, train_dl_scaled = evaluate_similarity(train_pred, train_ds)
        test_dl, test_dl_scaled = evaluate_similarity(test_pred, test_ds)
        train_sat = evaluate_satisfiability(tensor_dfa, train_pred)
        test_sat = evaluate_satisfiability(tensor_dfa, test_pred)
        return self.get_result_row(
            prefix, strategy, train_dl, train_dl_scaled, test_dl, test_dl_scaled, train_sat, test_sat
        )

    def get_result_row(self, prefix, strategy, train_dl, train_dl_scaled, test_dl, test_dl_scaled, train_sat, test_sat):
        return {
            'architecture': self.architecture,
            'dataset': self.dataset,
            'noise': self.noise,
            'alpha': self.alpha,
            'run_id': self.run_id,
            'model': self.model,
            'prefix_length': prefix,
            'sampling_strategy': strategy,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'train_distance': train_dl,
            'test_distance': test_dl,
            'train_similarity_scaled': train_dl_scaled,
            'test_similarity_scaled': test_dl_scaled,
            'train_satisfiability': train_sat,
            'test_satisfiability': test_sat,
            'nr_epochs': self.nr_epochs,
            'training_time': self.training_time
        }