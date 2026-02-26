from statistics import mean
import random
import numpy as np
from jellyfish import damerau_levenshtein_distance
import torch
import torch.nn.functional as F


def sample_token(logits, temperature=0, g=None):
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    else:
        probs = F.softmax(logits / temperature, dim=-1)
        probs = torch.clamp(probs, min=1e-10)
        return torch.multinomial(probs, num_samples=1, generator=g)


def sample(model, dataset, prefix_len, device, temperature=0, g=None):
    dataset = dataset.to(device)
    prefix = dataset[:, :prefix_len, :]
    predicted_traces = prefix.clone()

    stop_event = torch.zeros(dataset.size(-1), dtype=torch.float, device=device)
    stop_event_idx = dataset.size(-1) - 1
    stop_event[stop_event_idx] = 1.0

    stop_mask = torch.zeros(prefix.size(0), dtype=torch.bool).to(device)

    logits, rnn_state = model(prefix)

    for step in range(prefix_len, dataset.size(1)):
        logits_step = logits[:, -1, :]
        sample_idx = sample_token(logits_step, temperature, g)

        one_hot = F.one_hot(sample_idx.squeeze(-1), num_classes=logits_step.size(-1)).float().unsqueeze(1)
        predicted_traces = torch.cat((predicted_traces, one_hot), dim=1)

        stop_mask |= (sample_idx.squeeze(-1) == stop_event_idx)
        if torch.all(stop_mask):
            break

        logits, rnn_state = model.forward_from_state(one_hot, rnn_state)

    if not torch.all(stop_mask):
        stop_tensor = stop_event.unsqueeze(0).unsqueeze(1)
        stop_tensor = stop_tensor.expand(predicted_traces.size(0), -1, -1)
        predicted_traces = torch.cat((predicted_traces, stop_tensor), dim=1)

    return predicted_traces


def evaluate_similarity(predicted_traces, target_traces):
    dl_distances = []
    dl_similarities = []

    for i in range(predicted_traces.size()[0]):
        pred = tensor_to_string(predicted_traces[i])
        targ = tensor_to_string(target_traces[i])

        dl = damerau_levenshtein_distance(pred, targ)
        dl_scaled = 1 - (dl / max(len(pred), len(targ)))

        dl_distances.append(dl)
        dl_similarities.append(dl_scaled)

    return mean(dl_distances), mean(dl_similarities)


def tensor_to_string(one_hot_tensor):
    numpy_array = one_hot_tensor.cpu().numpy()
    stop_event = np.zeros(numpy_array.shape[-1])
    stop_event[-1] = 1

    string = ''
    for event in numpy_array:
        idx = event.argmax()
        string += chr(idx + 161)
        if np.array_equal(event, stop_event):
            break

    return string


def evaluate_satisfiability(dfa, predicted_traces):
    traces = torch.argmax(predicted_traces, dim=-1)

    r, _ = dfa(traces)
    accepted = r[:, -1, -1]

    return accepted.mean().item()