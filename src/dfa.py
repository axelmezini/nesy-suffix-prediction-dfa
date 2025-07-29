import re
import pydot
import networkx as nx
import torch
from torch import nn


class SymbolicDFA:
    def __init__(self, file_path, labels):
        self.file_path = file_path
        self.graph = nx.MultiDiGraph()
        self.initial_state = None
        self.state_types = {}
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.transition_table = {}
        self.parse_dot()

    def parse_dot(self):
        dot_graphs = pydot.graph_from_dot_file(self.file_path)
        dot = dot_graphs[0]

        for edge in dot.get_edges():
            src = edge.get_source()
            dst = edge.get_destination()
            if src == 'init':
                self.initial_state = int(dst)
            else:
                label = edge.get_label().strip('"')
                src, dst = int(src), int(dst)
                self.graph.add_edge(src, dst, key=label, label=label)

                label_idx = self.label_to_idx[label]
                self.transition_table.setdefault(src, {})[label_idx] = dst

        accepting_states = self.extract_accepting_states()
        rejecting_states = self.extract_rejecting_states(accepting_states)
        self.assign_state_types(accepting_states, rejecting_states)

    def extract_accepting_states(self):
        with open(self.file_path, 'r') as f:
            dot_source = f.read()

        accepting_states = set()
        double_circle_lines = re.findall(r'node\s*\[shape\s*=\s*doublecircle];\s*([\d\s;]+)', dot_source)
        for line in double_circle_lines:
            state_idx = line.split(';')[0]
            accepting_states.add(int(state_idx))
        return accepting_states

    def extract_rejecting_states(self, accepting_states):
        reachable = set()
        rev_graph = self.graph.reverse(copy=False)
        for accept in accepting_states:
            reachable.add(accept)
            visited = nx.descendants(rev_graph, accept)
            reachable.update(visited)
        return set(self.graph.nodes) - reachable

    def assign_state_types(self, accepting_states, rejecting_states):
        for state in self.graph.nodes:
            if state in accepting_states:
                self.state_types[state] = 1
            elif state in rejecting_states:
                self.state_types[state] = -1
            else:
                self.state_types[state] = 0


class TensorDFA():
    def __init__(self, dfa, device):
        super().__init__()
        self.dfa = dfa
        self.num_states = len(dfa.graph.nodes)
        self.vocab_size = len(dfa.label_to_idx)
        self.device = device
        self.transition_tensor = self.build_transition_tensor()
        self.state_types_tensor = torch.tensor([dfa.state_types[i] for i in range(self.num_states)], dtype=torch.long, device=self.device)

    def build_transition_tensor(self):
        table = torch.full((self.num_states, self.vocab_size), fill_value=-1, dtype=torch.long, device=self.device)
        for src, transitions in self.dfa.transition_table.items():
            for symbol_idx, dst in transitions.items():
                table[src, symbol_idx] = dst
        return table

    def simulate(self, input_indices, batch_size):
        seq_len = input_indices.size(1)
        current_states = torch.full((batch_size,), self.dfa.initial_state, dtype=torch.long, device=self.device)
        state_ids = []

        for t in range(seq_len):
            symbol = input_indices[:, t]
            next_states = self.transition_tensor[current_states, symbol].long()
            state_ids.append(next_states)
            current_states = next_states

        state_ids = torch.stack(state_ids, dim=1)
        return state_ids

    def check_satisfiability(self, traces_onehot):
        traces_indices = torch.argmax(traces_onehot, dim=-1)
        state_ids = self.simulate(traces_indices, traces_indices.size(0))
        final_states = state_ids[:, -1]
        is_accepted = (self.state_types_tensor[final_states] == 1).float()
        return is_accepted.mean().item()
