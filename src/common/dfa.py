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

        with open(str(self.folder_path / 'symbolicDFA.dot'), 'w+') as file:
            file.write(dot)

        self.build_from_file()

    def build_from_file(self):
        with open(str(self.folder_path / 'symbolicDFA.dot'), 'r') as file:
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

        with open(str(self.folder_path / 'simpleDFA_final.dot'), 'w+') as file:
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

    def forward(self, action_seq):
        batch_size, sequence_len, _ = action_seq.shape

        state = torch.zeros(batch_size, self.n_states, device=self.device)
        state[:, 0] = 1.0
        for i in range(sequence_len):
            state, reward = self.step(state, action_seq[:, i])
        return state, reward

    def step(self, state, action):
        selected_prob = state.unsqueeze(1).unsqueeze(-2) @ self.trans_prob
        next_state = action.unsqueeze(1) @ selected_prob.squeeze()
        next_reward = next_state @ self.accepting_matrix
        return next_state.squeeze(1), next_reward.squeeze(1)

    def unroll(self, action_seq):
        batch_size, seq_len, _ = action_seq.shape
        state = torch.zeros(batch_size, self.n_states, device=self.device)
        state[:, 0] = 1.0

        states = state.new_zeros(batch_size, seq_len, self.n_states)
        rewards = state.new_zeros(batch_size, seq_len, 2)

        for t in range(seq_len):
            state, reward = self.step(state, action_seq[:, t])
            states[:, t] = state
            rewards[:, t] = reward

        return states, rewards

    def next_states_rejecting(self, states):
        next_dist = torch.einsum('bts,asu->btau', states, self.trans_prob)
        reject_col = self.rejecting_matrix[:, 1]
        reject_prob = torch.einsum('btau,u->bta', next_dist, reject_col)
        return reject_prob > 0.5
