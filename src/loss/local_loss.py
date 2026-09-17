import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalLogicLoss(nn.Module):
    def __init__(self, dfa, alpha):
        super().__init__()
        self.dfa = dfa
        self.alpha = alpha

    def forward(self, predictions, targets, inputs):
        probs = F.softmax(predictions, dim=-1)
        batch_size, seq_len, vocab_size = predictions.shape

        states, _ = self.dfa.unroll(inputs)
        reject_mask = self.dfa.next_states_rejecting(states)

        gather_indices = targets.unsqueeze(-1)
        target_will_reject = torch.gather(reject_mask.float(), 2, gather_indices).squeeze(-1)  # (batch, seq_len)
        state_importance = (target_will_reject == 0).float()  # 1.0 when valid, 0.0 when rejecting

        ce_loss_per_step = F.cross_entropy(
            predictions.view(-1, vocab_size), targets.view(-1), reduction='none'
        ).view(batch_size, seq_len)
        weighted_ce_loss = (ce_loss_per_step * state_importance).sum() / (state_importance.sum() + 1e-6)

        invalid_mass = (probs * reject_mask.float()).sum(dim=-1)
        step_penalty = -torch.log(1.0 - invalid_mass + 1e-6).mean()

        return self.alpha * weighted_ce_loss + (1 - self.alpha) * step_penalty
