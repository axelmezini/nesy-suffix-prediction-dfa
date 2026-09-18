import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalLogicLoss(nn.Module):
    """
    Instead of rolling out full sequences, it only checks, at each ground-truth step, whether
    taking each possible next action would lead into a DFA-rejecting state,
    then combines a cross-entropy term (masking out steps whose true target
    itself leads to rejection) with a penalty on how much probability mass
    the model places on rejecting actions at each step. Alpha weights the
    tradeoff between the two terms.
    """
    def __init__(self, dfa, alpha):
        super().__init__()
        self.dfa = dfa
        self.alpha = alpha

    def forward(self, predictions, targets, inputs):
        probs = F.softmax(predictions, dim=-1)
        batch_size, seq_len, vocab_size = predictions.shape

        # For every step, get the DFA state distribution reached so far, and
        # for every possible next action, whether it would lead to a
        # rejecting state
        states = self.dfa.unroll(inputs)
        reject_mask = self.dfa.next_states_rejecting(states)

        # Look up, for the actual ground-truth target at each step, whether
        # taking it would have been rejecting; steps where the true target
        # leads to rejection are excluded from the cross-entropy term
        gather_indices = targets.unsqueeze(-1)
        target_will_reject = torch.gather(reject_mask.float(), 2, gather_indices).squeeze(-1)  # (batch, seq_len)
        state_importance = (target_will_reject == 0).float()  # 1.0 when valid, 0.0 when rejecting

        ce_loss_per_step = F.cross_entropy(
            predictions.view(-1, vocab_size), targets.view(-1), reduction='none'
        ).view(batch_size, seq_len)
        weighted_ce_loss = (ce_loss_per_step * state_importance).sum() / (state_importance.sum() + 1e-6)

        # Penalize probability mass the model assigns to any action that
        # would lead to a rejecting state, regardless of the ground truth
        invalid_mass = (probs * reject_mask.float()).sum(dim=-1)
        step_penalty = -torch.log(1.0 - invalid_mass + 1e-6).mean()

        return self.alpha * weighted_ce_loss + (1 - self.alpha) * step_penalty
