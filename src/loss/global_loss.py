import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLogicLoss(nn.Module):
    def __init__(self, model, dfa, alpha, prefixes, temperature=0.5, num_samples=10):
        super().__init__()
        self.model = model
        self.dfa = dfa
        self.prefixes = prefixes
        self.alpha = alpha
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(self, predictions, targets, inputs):
        prefix_len = random.choice(self.prefixes)
        prefix = inputs[:, :prefix_len, :]

        batch_size, _, num_activities = inputs.size()

        end_mask = inputs[:, :, -1] == 1
        first_end_idx = end_mask.float().argmax(dim=1)
        no_end_mask = ~end_mask.any(dim=1)
        first_end_idx[no_end_mask] = inputs.size(1)
        max_truncated_length = first_end_idx.max().item()

        prefix = prefix.unsqueeze(1).repeat(1, self.num_samples, 1, 1).view(-1, prefix_len, num_activities)

        next_event, rnn_state = self.model(prefix)
        dfa_state, dfa_rew = self.dfa.forward(prefix)

        for step in range(prefix_len, max_truncated_length + 10):
            next_event = F.log_softmax(next_event[:, -1:, :], dim=-1)
            next_event_one_hot = gumbel_softmax(next_event, self.temperature)

            dfa_state, dfa_rew = self.dfa.step_pi(dfa_state, next_event_one_hot.squeeze())
            next_event, rnn_state = self.model.forward_from_state(next_event_one_hot, rnn_state)

        dfa_rew = dfa_rew.view(batch_size, self.num_samples, 2)
        dfa_rew = dfa_rew[:, :, 1]

        log_loss = -torch.log(torch.mean(dfa_rew, dim=-1).clamp(min=1e-10)).mean()
        sup_loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1))
        return self.alpha * sup_loss + (1 - self.alpha) * log_loss


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)