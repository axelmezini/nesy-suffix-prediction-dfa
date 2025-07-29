from statistics import mean
from torch.utils.data import TensorDataset, DataLoader
import torch.nn
import torchmetrics
import torch.nn.functional as F
from modules import LogicLoss

MAX_NUM_EPOCHS = 2000
BATCH_SIZE = 64


class EarlyStopping:
    def __init__(self, patience=25, min_delta=1e-5, min_loss=0.01):
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


def train(model, dfa, train_dataset, test_dataset, device):
    loss_func = torch.nn.CrossEntropyLoss()
    logic_loss = LogicLoss(dfa)
    optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
    acc_func = torchmetrics.Accuracy(task='multiclass', num_classes=train_dataset.size(-1), top_k=1).to(device)
    early_stopper = EarlyStopping()
    train_losses, test_losses = [], []

    X_data = train_dataset[:, :-1, :]
    Y_data = train_dataset[:, 1:, :]
    train_loader = DataLoader(TensorDataset(X_data, Y_data), batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(MAX_NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, dfa, train_loader, acc_func, logic_loss, optim, device)
        test_loss, test_acc = test(model, test_dataset, acc_func, loss_func, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}:\ttrain loss: {train_loss:.4f}\ttest loss: {test_loss:.4f}\ttrain acc: {train_acc:.4f}\ttest acc: {test_acc:.4f}")

        if epoch >= 100 and early_stopper(train_loss):
            return train_acc, test_acc, train_losses, test_losses, epoch

    return train_acc, test_acc, train_losses, test_losses, epoch


def train_epoch(model, dfa, train_loader, acc_func, logic_loss, optim, device):
    model.train()
    batch_accuracies, batch_losses = [], []

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        vocab_size = X.size(-1)
        targets = torch.argmax(Y, dim=-1)
        optim.zero_grad()

        predictions, _ = model(X)

        if dfa:
            total_loss = logic_loss(predictions, targets, X, device)
        else:
            total_loss = F.cross_entropy(predictions.view(-1, vocab_size), targets.view(-1))

        total_loss.backward()
        optim.step()

        batch_losses.append(total_loss.item())
        acc = acc_func(predictions.view(-1, vocab_size), targets.view(-1)).item()
        batch_accuracies.append(acc)

    return mean(batch_losses), mean(batch_accuracies)


def test(model, test_dataset, acc_func, loss_func, device):
    model.eval()
    accuracies = []
    losses = []

    X_data = test_dataset[:, :-1, :]
    Y_data = test_dataset[:, 1:, :]
    test_tensor_dataset = TensorDataset(X_data, Y_data)
    test_loader = DataLoader(test_tensor_dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)

            target = torch.argmax(Y.reshape(-1, Y.size(-1)), dim=-1)
            predictions, _ = model(X)
            predictions = predictions.reshape(-1, predictions.size(-1))

            loss = loss_func(predictions, target)
            losses.append(loss.item())
            accuracies.append(acc_func(predictions, target).item())

    return mean(losses), mean(accuracies)
