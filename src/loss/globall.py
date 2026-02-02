import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLogicLoss(nn.Module):
    def __init__(self, model, dfa, alpha, device, prefixes, temperature=0.5, num_samples=10):
        super().__init__()
        self.model = model
        self.dfa = dfa
        self.prefixes = prefixes
        self.alpha = alpha
        self.temperature = temperature
        self.device = device
        self.num_samples = num_samples

    def forward(self, predictions, targets, inputs):
        dataset = inputs.to(self.device)
        prefix_len = random.choice(self.prefixes)

        prefix = dataset[:, :prefix_len, :]

        batch_size, len_traces, num_activities = dataset.size()

        # -----
        end_mask = dataset[:, :, -1] == 1
        first_end_idx = end_mask.float().argmax(dim=1)
        no_end_mask = ~end_mask.any(dim=1)
        first_end_idx[no_end_mask] = dataset.size(1)
        max_truncated_length = first_end_idx.max().item()
        # -----

        # target = torch.ones(batch_size, dtype=torch.long, device=device)

        # extend prefix
        prefix = prefix.unsqueeze(1).repeat(1, self.num_samples, 1, 1).view(-1, prefix_len, num_activities)

        # calculate next symbol and dfa state
        next_event, rnn_state = self.model(prefix)
        dfa_states, dfa_rew = self.dfa.forward_pi(prefix)
        dfa_state = dfa_states[:, -1, :]

        log_prob_traces = torch.zeros((batch_size * self.num_samples, 1)).to(self.device)
        for step in range(prefix_len, max_truncated_length + 10):
            # next_event = next_event[:, -1:, :]
            next_event = F.log_softmax(next_event[:, -1:, :], dim=-1)
            next_event_one_hot = gumbel_softmax(next_event, self.temperature)

            log_prob_traces += torch.sum(next_event * next_event_one_hot, dim=-1)
            # transit on the automaton
            dfa_state, dfa_rew = self.dfa.step_pi(dfa_state, next_event_one_hot.squeeze())
            # transit the rnn
            next_event, rnn_state = self.model.forward_from_state(next_event_one_hot, rnn_state)

        dfa_rew = dfa_rew.view(batch_size, self.num_samples, 2)
        dfa_rew = dfa_rew[:, :, 1]

        # log_prob_traces = log_prob_traces.view(batch_size, self.num_samples)
        # prob_acceptance = torch.sum(torch.exp(log_prob_traces) * dfa_rew, dim=-1)
        # log_loss = -torch.log(prob_acceptance.clamp(min=1e-10)).mean()
        # p = prob_acceptance.mean().item()
        # p = max(0.0, min(1.0, p))
        # deviation = 1.96 * sqrt(p * (1 - p) / num_samples)

        ## prob_acceptance = torch.sum(torch.nn.functional.softmax(log_prob_traces, dim=-1) * dfa_rew, dim=-1)

        log_loss = -torch.log(torch.mean(dfa_rew, dim=-1).clamp(min=1e-10)).mean()

        return log_loss#, deviation


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)