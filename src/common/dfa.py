import os.path
from sympy import sympify, symbols
import pydot
import networkx as nx
from ltlf2dfa.parser.ltlf import LTLfParser
import torch
from torch import nn


class SymbolicDFA:
    def __init__(self, labels, folder_path):
        self.labels = labels
        self.folder_path = folder_path
        self.graph = nx.MultiDiGraph()
        self.initial_state = None
        self.accepting_state = None
        self.state_types = {}

    def build_from_formula(self, formula):
        parser = LTLfParser()
        ast = parser(formula)
        dot = ast.to_dfa()

        with open(os.path.join(self.folder_path, 'symbolicDFA.dot'), 'w+') as file:
            file.write(dot)

        self.build_from_file()

    def build_from_file(self):
        with open(os.path.join(self.folder_path, 'symbolicDFA.dot'), 'r') as file:
            dot = file.read()

        token_symbols = symbols(self.labels)
        token_map = dict(zip(self.labels, token_symbols))

        temp_accepting_states = []
        for line in dot.splitlines():
            if 'doublecircle' in line:
                finals = line.strip().split(';')[1:-1]
                temp_accepting_states = [int(s.strip()) - 1 for s in finals]
            elif '->' in line:
                if 'init' in line:
                    parts = line.strip().split(' ')
                    self.initial_state = int(parts[2][:-1]) - 1
                else:
                    parts = line.strip().split(' ')
                    src, dst = int(parts[0]) - 1, int(parts[2]) - 1
                    label = line.strip().split('"')[1]

                    guard = sympify(a=label, locals=token_map)
                    for token in valid_tokens_for_guard(guard, self.labels):
                        self.graph.add_edge(src, dst, token)

        initial_states = list(self.graph.nodes)
        self.accepting_state = max(self.graph.nodes) + 1
        final_rejecting = max(self.graph.nodes) + 2
        self.graph.add_node(self.accepting_state)

        for state in initial_states:
            if state in temp_accepting_states:
                self.graph.add_edge(state, self.accepting_state, 'end')
            else:
                self.graph.add_edge(state, final_rejecting, 'end')

        all_rejecting = self.extract_rejecting_states()
        for state in self.graph.nodes:
            if state == self.accepting_state:
                self.state_types[state] = 1
            elif state in all_rejecting:
                self.state_types[state] = -1
            else:
                self.state_types[state] = 0

        for label in self.labels + ['end']:
            self.graph.add_edge(final_rejecting, final_rejecting, label)
            self.graph.add_edge(self.accepting_state, self.accepting_state, label)

        self.write_final_dot_to_file()

    def extract_rejecting_states(self):
        rev_graph = self.graph.reverse(copy=False)
        reachable = {self.accepting_state}
        reachable.update(nx.descendants(rev_graph, self.accepting_state))
        return self.graph.nodes - reachable

    def to_deep_dfa(self, device):
        deep_dfa = DeepDFA(len(self.graph.nodes), len(self.labels) + 1, device)
        deep_dfa.build(self.state_types, self.graph.edges, self.labels)
        return deep_dfa

    def write_final_dot_to_file(self):
        intro = """digraph MONA_DFA {
rankdir = LR;
center = true;
size = "7.5,10.5";
edge [fontname = Courier];
node [height = .5, width = .5];
"""
        end = f'node [shape = doublecircle]; {self.accepting_state};'
        start = f'node [shape = circle]; {self.initial_state};\ninit [shape = plaintext, label = ""];\ninit -> {self.initial_state};'
        transitions_string = ""
        for src, dst, label in self.graph.edges:
            transitions_string += f'{src} -> {dst} [label="{label}"];\n'
        transitions_string += "}"

        with open(os.path.join(self.folder_path, 'simpleDFA_final.dot'), 'w+') as file:
            file.write(intro + end + '\n' + start + '\n' + transitions_string)


def valid_tokens_for_guard(guard_expr, tokens):
    valid = []
    for token in tokens:
        assignment = {t: False for t in tokens}
        assignment[token] = True
        if bool(guard_expr.subs(assignment)):
            valid.append(token)
    return valid


class DeepDFA(nn.Module):
    def __init__(self, n_states, n_actions, device):
        super(DeepDFA, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.device = device
        self.trans_prob = torch.zeros((n_actions, n_states, n_states), requires_grad=False, device=device)
        self.accepting_matrix = torch.zeros((n_states, 2), requires_grad=False, device=device)
        self.rejecting_matrix = torch.zeros((n_states, 2), requires_grad=False, device=device)

    def build(self, state_types, edges, labels):
        labels_map = {label: i for i, label in enumerate(labels + ['end'])}

        with torch.no_grad():
            for (src, dst, label) in edges:
                self.trans_prob[labels_map[label], src, dst] = 1.0

            for s in state_types:
                self.accepting_matrix[s, int(state_types[s] == 1)] = 1.0
                self.rejecting_matrix[s, int(state_types[s] == -1)] = 1.0

    def forward(self, action_seq, current_state=None):
        torch.set_printoptions(threshold=torch.inf)
        batch_size = action_seq.size()[0]
        length_seq = action_seq.size()[1]
        rewards = torch.zeros((batch_size, length_seq, 2)).to(self.device)  # pylint: disable=no-member
        if current_state is None:
            s = torch.zeros((batch_size, self.n_states)).to(self.device)  # pylint: disable=no-member
            # initial state is 0 for construction
            s[:, 0] = 1.0
        else:
            s = current_state

        for i in range(length_seq):
            a = action_seq[:, i]
            s, r = self.step(s, a)
            rewards[:, i:i+1, :] = r.unsqueeze(1).expand(-1, 1, -1)
        return rewards, s

    def step(self, state, action, verb=False):
        if isinstance(action, int):
            action = torch.IntTensor([action])
        if verb:
            print(action)
        selected_prob = torch.index_select(self.trans_prob, 0, action)  # pylint: disable=no-member
        next_state = torch.matmul(state.unsqueeze(dim=1), selected_prob)  # pylint: disable=no-member

        next_output = torch.matmul(next_state, self.accepting_matrix)  # pylint: disable=no-member
        next_state = next_state.squeeze(1)
        next_output = next_output.squeeze(1)
        return next_state, next_output

    def forward_pi(self, action_seq, current_state=None):
        batch_size = action_seq.size()[0]
        length_size = action_seq.size()[1]

        pred_states = torch.zeros((batch_size, length_size, self.n_states)).to(self.device)  # pylint: disable=no-member
        pred_rew = torch.zeros((batch_size, length_size, 2)).to(self.device)  # pylint: disable=no-member

        if current_state is None:
            s = torch.zeros((batch_size, self.n_states)).to(self.device)  # pylint: disable=no-member
            # initial state is 0 for construction
            s[:, 0] = 1.0
        else:
            s = current_state
        for i in range(length_size):
            a = action_seq[:, i, :]

            s, r = self.step_pi(s, a)

            pred_states[:, i, :] = s
            pred_rew[:, i, :] = r

        return pred_states, pred_rew

    def step_pi(self, state, action):
        # no activation
        trans_prob = self.trans_prob
        rew_matrix = self.accepting_matrix

        trans_prob = trans_prob.unsqueeze(0)
        state = state.unsqueeze(1).unsqueeze(-2)

        selected_prob = torch.matmul(state, trans_prob)  # pylint: disable=no-member

        next_state = torch.matmul(action.unsqueeze(1), selected_prob.squeeze())  # pylint: disable=no-member
        next_reward = torch.matmul(next_state, rew_matrix)  # pylint: disable=no-member

        return next_state.squeeze(1), next_reward.squeeze(1)

    def simulate(self, action_seq):
        batch_size = action_seq.size()[0]
        length_seq = action_seq.size()[1]

        trans_argmax = torch.argmax(self.trans_prob, dim=2).transpose(0, 1).to(torch.long).to(self.device)
        state_ids = torch.zeros((batch_size, length_seq), dtype=torch.long, device=self.device)
        current = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # start at state 0
        for t in range(length_seq):
            symbols = action_seq[:, t]
            current = trans_argmax[current, symbols]
            state_ids[:, t] = current
        return state_ids

    def next_states_rejecting(self, state_ids):
        # state_ids: (batch, seq_len) long, representing state BEFORE consuming next symbol
        batch_size, seq_len = state_ids.shape
        device = state_ids.device
        trans_argmax = torch.argmax(self.trans_prob, dim=2).transpose(0, 1).to(torch.long).to(
            device)  # (n_states, n_actions)

        flat = state_ids.view(-1)  # (batch*seq,)
        next_states_flat = trans_argmax[flat]  # (batch*seq, n_actions)
        next_states = next_states_flat.view(batch_size, seq_len, self.n_actions)  # (batch, seq, n_actions)

        # rejecting_matrix has 1 in column 1 for rejecting states
        # gather column 1 for next_states
        reject_col = self.rejecting_matrix[:, 1].to(device)  # (n_states,)
        reject_mask = reject_col[next_states] == 1  # bool (batch, seq, n_actions)
        return reject_mask