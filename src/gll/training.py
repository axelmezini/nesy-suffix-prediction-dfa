import torch
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
from statistics import mean
from gll.evaluation import evaluate_accuracy_next_activity, logic_loss_multiple_samples


if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def train_old(rnn, train_dataset, test_dataset, max_num_epochs, alpha, deepdfa=None, prefixes=[], batch_size=64):
    curr_temp = 0.5
    lambda_temp = 0.999
    min_temp = 0.0001

    rnn = rnn.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=rnn.parameters(), lr=0.0005)
    acc_func = torchmetrics.Accuracy(task='multiclass', num_classes=train_dataset.size(-1), top_k=1).to(device)
    sup_stopper = EarlyStopping()

    sup_losses = []
    log_losses = []
    deviations = []

    X_data = train_dataset[:, :-1, :]
    Y_data = train_dataset[:, 1:, :]
    train_tensor_dataset = TensorDataset(X_data, Y_data)
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(max_num_epochs):
        train_acc_batches = []
        sup_loss_batches = []
        log_loss_batches = []
        deviations_batches = []

        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)

            target = torch.argmax(Y.reshape(-1, Y.size(-1)), dim=-1)
            optim.zero_grad()

            predictions, _ = rnn(X)
            predictions = predictions.reshape(-1, predictions.size(-1))

            sup_loss = loss_func(predictions, target)
            sup_loss_batches.append(sup_loss.item())

            if deepdfa is not None:
                log_loss, deviation = logic_loss_multiple_samples(rnn, deepdfa, X, prefixes, curr_temp, num_samples=10)
                log_loss_batches.append(log_loss.item())
                deviations_batches.append(deviation)
                loss = alpha * sup_loss + (1-alpha) * log_loss
            else:
                loss = sup_loss

            loss.backward()
            optim.step()

            train_acc_batches.append(acc_func(predictions, target).item())

        train_acc = mean(train_acc_batches)
        test_acc = evaluate_accuracy_next_activity(rnn, test_dataset, acc_func)

        sup_loss_epoch = mean(sup_loss_batches)
        sup_losses.append(sup_loss_epoch)

        if deepdfa is not None:
            log_loss_epoch = mean(log_loss_batches)
            log_losses.append(log_loss_epoch)
            deviations.append(mean(deviations_batches))
            epoch_loss = 0.75 * sup_loss_epoch + 0.25 * log_loss_epoch
            if epoch % 100 == 0:
                #print(f"Epoch {epoch}:\tloss: {sup_loss_epoch:.4f}\tlogic_loss: {log_loss_epoch:.4f}\ttrain acc: {train_acc:.4f}\ttest acc: {test_acc:.4f}")
                print(f"Epoch {epoch}:\tsup_loss: {sup_loss_epoch:.8f}\tlogic_loss: {log_loss_epoch:.8f}")
                print(f"Epoch {epoch}:\ttrain loss: {epoch_loss:.8f}")
            if epoch >= 500 and sup_stopper(epoch_loss):
                return train_acc, test_acc, sup_losses, log_losses, deviations, epoch
        else:
            if epoch % 100 == 0:
                print(f"Epoch {epoch}:\tloss: {sup_loss_epoch:.4f}\ttrain acc: {train_acc:.4f}\ttest acc: {test_acc:.4f}")
            if epoch >= 500 and sup_stopper(sup_loss_epoch):
                return train_acc, test_acc, sup_losses, log_losses, deviations, epoch

        #curr_temp = max(lambda_temp * curr_temp, min_temp)

    return train_acc, test_acc, sup_losses, log_losses, deviations, epoch


class EarlyStopping:
    def __init__(self, patience=35, min_delta=1e-5, min_loss=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = min_loss
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, new_loss):
        if new_loss < self.best_loss - self.min_delta:
            self.best_loss = new_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience or new_loss < self.min_loss
