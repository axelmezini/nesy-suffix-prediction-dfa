import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LocalLogicLoss(nn.Module):
    def __init__(self, dfa, alpha, device):
        super().__init__()
        self.dfa = dfa
        self.alpha = alpha
        self.device = device

    def forward(self, predictions, targets, inputs):
        probs = F.softmax(predictions, dim=-1)
        batch_size, seq_len, vocab_size = predictions.shape

        # DFA transitions
        token_indices = torch.argmax(inputs, dim=-1)
        state_ids = self.dfa.simulate(token_indices)

        reject_mask = self.dfa.next_states_rejecting(state_ids)

        # Cross-Entropy Loss per step
        ce_loss_per_step = F.cross_entropy(
            predictions.view(-1, vocab_size), targets.view(-1), reduction='none'
        ).view(batch_size, seq_len)

        gather_indices = targets.unsqueeze(-1)
        target_will_reject = torch.gather(reject_mask.float(), 2, gather_indices).squeeze(-1)  # (batch, seq_len)
        state_importance = (target_will_reject == 0).float()  # 1.0 when valid, 0.0 when rejecting

        ce_weights = 1 * state_importance
        weighted_ce_loss = (ce_loss_per_step * ce_weights).sum() / (ce_weights.sum() + 1e-6)

        # Symbolic Loss
        invalid_mass = (probs * reject_mask.float()).sum(dim=-1).mean()
        step_penalty = -torch.log(1.0 - invalid_mass + 1e-6)

        return self.alpha * weighted_ce_loss + (1 - self.alpha) * step_penalty


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