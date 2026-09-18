import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLogicLoss(nn.Module):
    """
    Combines standard supervised next-event loss with a "global" logic loss
    that rolls the model out autoregressively (differentiably, via Gumbel-
    softmax) from a random prefix to the end of the trace, and penalizes it
    based on how well the *generated continuation* satisfies the DFA.
    Alpha weights the tradeoff between the two terms.
    """
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

        # Find where each trace's "end" event occurs (or its full length if
        # it never ends) to know how far the longest trace needs to be rolled out
        end_mask = inputs[:, :, -1] == 1
        first_end_idx = end_mask.float().argmax(dim=1)
        no_end_mask = ~end_mask.any(dim=1)
        first_end_idx[no_end_mask] = inputs.size(1)
        max_truncated_length = first_end_idx.max().item()

        # Draw num_samples independent continuations per trace by repeating
        # each prefix along a new "samples" axis, folded into the batch dim
        prefix = prefix.unsqueeze(1).repeat(1, self.num_samples, 1, 1).view(-1, prefix_len, num_activities)

        next_event, rnn_state = self.model(prefix)
        dfa_state, _ = self.dfa(prefix)

        # Autoregressive rollout: at each step, turn the model's logits into
        # a differentiable ("soft") one-hot event via Gumbel-softmax, advance the DFA
        # state with that soft event, and feed it back into the model
        for step in range(prefix_len, max_truncated_length + 10):
            next_event = F.log_softmax(next_event[:, -1:, :], dim=-1)
            next_event_one_hot = gumbel_softmax(next_event, self.temperature)

            dfa_state = self.dfa.step(dfa_state, next_event_one_hot.squeeze())
            next_event, rnn_state = self.model.forward_from_state(next_event_one_hot, rnn_state)

        # Reshape rewards back to (batch, samples) and keep only the
        # "accepting" reward column; average acceptance probability across
        # samples per trace, then use -log(...) as the logic loss (low when
        # the DFA is satisfied with high probability)
        dfa_rew = dfa_state @ self.dfa.accepting_matrix
        dfa_rew = dfa_rew.view(batch_size, self.num_samples, 2)[:, :, 1]

        log_loss = -torch.log(torch.mean(dfa_rew, dim=-1).clamp(min=1e-10)).mean()
        sup_loss = F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1))
        return self.alpha * sup_loss + (1 - self.alpha) * log_loss


def gumbel_softmax(logits, temperature=1.0, eps=1e-10):
    # Adds Gumbel noise to log-probabilities and applies a temperature-scaled
    # softmax, giving a differentiable approximation of sampling a discrete
    # category (lower temperature -> closer to a true one-hot sample)
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)
