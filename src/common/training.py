from statistics import mean
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torchmetrics
import torch.nn.functional as F


class EarlyStopping:
    def __init__(self, patience, min_delta, min_loss):
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


def train(architecture, train_dataset, test_dataset, config, loss, loss_fn):
    device = config.device
    optim = torch.optim.Adam(params=architecture.parameters(), lr=config.lr)
    acc_func = torchmetrics.Accuracy(task='multiclass', num_classes=train_dataset.size(-1), top_k=1).to(device)
    early_stopper = EarlyStopping(config.patience, config.min_delta, config.min_loss)

    X_data = train_dataset[:, :-1, :]
    Y_data = train_dataset[:, 1:, :]
    train_loader = DataLoader(TensorDataset(X_data, Y_data), batch_size=config.batch_size, shuffle=False)

    print(type(loss_fn))
    for epoch in range(config.nr_epochs):
        train_loss, train_acc = train_epoch(architecture, train_loader, acc_func, loss_fn, optim, device, loss)
        test_loss, test_acc = test(architecture, test_dataset, acc_func, device, config.batch_size)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}:\ttotal loss: {train_loss:.8f}")

        if epoch >= config.min_epochs and early_stopper(train_loss):
            return train_acc, test_acc, epoch

    return train_acc, test_acc, epoch


def train_epoch(architecture, train_loader, acc_func, loss_fn, optim, device, loss):
    batch_accuracies, batch_losses = [], []

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        vocab_size = X.size(-1)
        targets = torch.argmax(Y, dim=-1)
        optim.zero_grad()

        predictions, _ = architecture(X)
        if loss == 'baseline':
            total_loss = F.cross_entropy(predictions.reshape(-1, vocab_size), targets.reshape(-1))
        else:
            total_loss = loss_fn(predictions, targets, X)

        total_loss.backward()
        optim.step()

        batch_losses.append(total_loss.item())
        acc = acc_func(predictions.view(-1, vocab_size), targets.view(-1)).item()
        batch_accuracies.append(acc)

    return mean(batch_losses), mean(batch_accuracies)


def test(architecture, test_dataset, acc_func, device, batch_size):
    accuracies,  losses = [], []

    X_data = test_dataset[:, :-1, :]
    Y_data = test_dataset[:, 1:, :]
    test_loader = DataLoader(TensorDataset(X_data, Y_data), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)

            predictions, _ = architecture(X)
            predictions = predictions.reshape(-1, predictions.size(-1))
            target = torch.argmax(Y.reshape(-1, Y.size(-1)), dim=-1)

            loss = F.cross_entropy(predictions, target)
            losses.append(loss.item())
            accuracies.append(acc_func(predictions, target).item())

    return mean(losses), mean(accuracies)
