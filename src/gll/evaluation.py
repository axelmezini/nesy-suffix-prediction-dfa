import random
import torch.nn.functional as F
import torch
from math import sqrt
import numpy as np
from statistics import mean
from jellyfish import damerau_levenshtein_distance

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

cross_entr_func = torch.nn.CrossEntropyLoss()

def evaluate_accuracy_next_activity(rnn, test_dataset, acc_func):
    rnn = rnn.to(device)
    accuracies = []
    for batch in [test_dataset]:
        X = batch[:, :-1, :].to(device)
        Y = batch[:, 1:, :]
        target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)
        # print(target.size())
        with torch.no_grad():
            predictions, _ = rnn(X)
        predictions = predictions.reshape(-1, predictions.size()[-1])

        accuracies.append(acc_func(predictions, target).item())

    return mean(accuracies)


def sample_with_temperature(logits, temperature=1.0, g=None):
    if temperature == 0:
        indices = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        indices = torch.multinomial(probs, num_samples=1, generator=g)

    batch_size = logits.size(0)
    num_classes = logits.size(-1)
    one_hot = torch.zeros(batch_size, 1, num_classes).to(device)
    one_hot.scatter_(2, indices.unsqueeze(-1), 1)
    return one_hot, indices


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def logic_loss_multiple_samples(rnn, deepdfa, data, prefixes, temperature=1.0, num_samples=10):
    dataset = data.to(device)
    prefix_len = random.choice(prefixes)
    prefix = dataset[:, :prefix_len, :]

    batch_size, len_traces, num_activities = dataset.size()

    #-----
    end_mask = dataset[:, :, -1] == 1
    first_end_idx = end_mask.float().argmax(dim=1)
    no_end_mask = ~end_mask.any(dim=1)
    first_end_idx[no_end_mask] = dataset.size(1)
    max_truncated_length = first_end_idx.max().item()
    #-----

    prefix = prefix.unsqueeze(1).repeat(1, num_samples, 1, 1).view(-1, prefix_len, num_activities)

    next_event, rnn_state = rnn(prefix)
    dfa_states, dfa_rew = deepdfa.forward_pi(prefix)
    dfa_state = dfa_states[:, -1, :]

    log_prob_traces = torch.zeros((batch_size*num_samples, 1)).to(device)
    for step in range(prefix_len, max_truncated_length + 10):
        next_event = F.log_softmax(next_event[:, -1:, :], dim=-1)
        next_event_one_hot = gumbel_softmax(next_event, temperature)

        log_prob_traces += torch.sum(next_event * next_event_one_hot, dim=-1)
        dfa_state, dfa_rew = deepdfa.step_pi(dfa_state, next_event_one_hot.squeeze())
        next_event, rnn_state = rnn.forward_from_state(next_event_one_hot, rnn_state)

    dfa_rew = dfa_rew.view(batch_size, num_samples, 2)
    dfa_rew = dfa_rew[:, :, 1]
    loss = -torch.log(torch.mean(dfa_rew, dim=-1).clamp(min=1e-10)).mean()

    return loss, 0


def suffix_prediction_with_temperature_with_stop(model, dataset, prefix_len, stop_event, temperature=1.0, g=None):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    predicted_traces = prefix.clone()

    logits, rnn_state = model(prefix)
    stop_event_idx = stop_event.index(1)
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, dataset.size(1)):
        logits_step = logits[:, -1, :]
        one_hot, sample_idx = sample_with_temperature(logits_step, temperature, g)

        predicted_traces = torch.cat((predicted_traces, one_hot), dim=1)

        stop_mask |= (sample_idx.squeeze(-1) == stop_event_idx)
        if torch.all(stop_mask):
            break

        logits, rnn_state = model.forward_from_state(one_hot, rnn_state)

    if not torch.all(stop_mask):
        stop_tensor = torch.tensor(stop_event, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(1)
        stop_tensor = stop_tensor.expand(predicted_traces.size(0), -1, -1)
        predicted_traces = torch.cat((predicted_traces, stop_tensor), dim=1)

    return predicted_traces


def greedy_suffix_prediction_with_stop(model, dataset, prefix_len, stop_event):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    predicted_traces = prefix.clone()

    logits, rnn_state = model(prefix)
    stop_event_idx = stop_event.index(1)
    stop_mask = torch.zeros(prefix.size(0)).bool().to(device)

    for step in range(prefix_len, dataset.size(1)):
        logits_step = logits[:, -1, :]

        top_idx = torch.argmax(logits_step, dim=-1, keepdim=True)
        one_hot = F.one_hot(top_idx.squeeze(-1), num_classes=logits_step.size(-1)).float().unsqueeze(1)
        predicted_traces = torch.cat((predicted_traces, one_hot), dim=1)
        stop_mask |= (top_idx.squeeze(-1) == stop_event_idx)

        if torch.all(stop_mask):
            break

        logits, rnn_state = model.forward_from_state(one_hot, rnn_state)

    if not torch.all(stop_mask):
        stop_tensor = torch.tensor(stop_event, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(1)
        stop_tensor = stop_tensor.expand(predicted_traces.size(0), -1, -1)
        predicted_traces = torch.cat((predicted_traces, stop_tensor), dim=1)

    return predicted_traces
