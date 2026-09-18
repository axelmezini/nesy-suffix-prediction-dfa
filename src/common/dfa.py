import os.path
from sympy import sympify, symbols
import pydot
import networkx as nx
from ltlf2dfa.parser.ltlf import LTLfParser
import torch
from torch import nn


class SymbolicDFA:
    """
    Builds a symbolic DFA from an LTLf formula, normalizes it into a single accepting sink state and
    a single rejecting sink state, and can convert it into a differentiable
    DeepDFA for use as a training signal
    """
    def __init__(self, labels, folder_path):
        self.labels = labels
        self.folder_path = folder_path
        self.graph = nx.MultiDiGraph()
        self.initial_state = None
        self.accepting_state = None
        self.state_types = {}

    def build_from_formula(self, formula):
        """
        Converts an LTLf formula into a DFA via ltlf2dfa, caches the raw
        MONA-style .dot output to disk, then parses it into the graph
        """
        parser = LTLfParser()
        ast = parser(formula)
        dot = ast.to_dfa()

        with open(str(self.folder_path / 'symbolicDFA.dot'), 'w+') as file:
            file.write(dot)

        self.build_from_file()

    def build_from_file(self):
        """
        Parses the cached MONA .dot file into self.graph: finds the
        accepting states and initial state, and for each transition
        evaluates its boolean guard formula against every possible single
        active label to expand it into concrete labeled edges. Then adds a
        single shared accepting sink and rejecting sink state (reached via
        an 'end' event), classifies every state as accepting/rejecting/
        neutral, and makes both sink states self-looping on every label so
        sequences can run indefinitely from them.
        """
        with open(str(self.folder_path / 'symbolicDFA.dot'), 'r') as file:
            dot = file.read()

        token_symbols = symbols(self.labels)
        token_map = dict(zip(self.labels, token_symbols))

        temp_accepting_states = []
        for line in dot.splitlines():
            if 'doublecircle' in line:
                # MONA marks accepting states with a doublecircle node shape line
                finals = line.strip().split(';')[1:-1]
                temp_accepting_states = [int(s.strip()) - 1 for s in finals]
            elif '->' in line:
                if 'init' in line:
                    # The synthetic "init -> N" edge marks the real initial state
                    parts = line.strip().split(' ')
                    self.initial_state = int(parts[2][:-1]) - 1
                else:
                    parts = line.strip().split(' ')
                    src, dst = int(parts[0]) - 1, int(parts[2]) - 1
                    label = line.strip().split('"')[1]

                    # Each dot edge label is a boolean guard over all activity
                    # symbols; expand it into one edge per concrete label that satisfies the guard
                    guard = sympify(a=label, locals=token_map)
                    for token in valid_tokens_for_guard(guard, self.labels):
                        self.graph.add_edge(src, dst, token)

        initial_states = list(self.graph.nodes)
        self.accepting_state = max(self.graph.nodes) + 1
        final_rejecting = max(self.graph.nodes) + 2
        self.graph.add_node(self.accepting_state)

        # Every original state gets an 'end' transition into either the
        # single shared accepting sink or the single shared rejecting sink
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

        # Make both sinks absorbing: every label (including 'end') loops back to itself
        for label in self.labels + ['end']:
            self.graph.add_edge(final_rejecting, final_rejecting, label)
            self.graph.add_edge(self.accepting_state, self.accepting_state, label)

        self.write_final_dot_to_file()

    def extract_rejecting_states(self):
        """
        A state is "rejecting" if the accepting state is not reachable from it;
        found by reversing the graph and taking all nodes that can't reach the
        accepting state via a reverse-reachability search from it
        """
        rev_graph = self.graph.reverse(copy=False)
        reachable = {self.accepting_state}
        reachable.update(nx.descendants(rev_graph, self.accepting_state))
        return self.graph.nodes - reachable

    def to_deep_dfa(self, device):
        """
        Converts this DFA into a tensor-based DeepDFA usable in training
        """
        deep_dfa = DeepDFA(len(self.graph.nodes), len(self.labels) + 1).to(device)
        deep_dfa.build(self.state_types, self.graph.edges, self.labels)
        return deep_dfa

    def write_final_dot_to_file(self):
        """
        Writes the normalized graph (with the merged accepting/rejecting
        sinks) back out as a MONA-style .dot file
        """
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
    """
    For each token, tests the boolean guard with only that token set True
    and keeps it if the guard evaluates to true
    """
    valid = []
    for token in tokens:
        assignment = {t: False for t in tokens}
        assignment[token] = True
        if bool(guard_expr.subs(assignment)):
            valid.append(token)
    return valid


class DeepDFA(nn.Module):
    """
    A differentiable, tensor-based representation of a DFA: state is a
    probability distribution over DFA states, and transitions are applied
    via matrix multiplication with a one-hot transition tensor, so it can
    be used as part of a gradient-based loss
    """
    def __init__(self, n_states, n_actions):
        super(DeepDFA, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.register_buffer('trans_prob', torch.zeros(n_actions, n_states, n_states))
        self.register_buffer('accepting_matrix', torch.zeros(n_states, 2))
        self.register_buffer('rejecting_matrix', torch.zeros(n_states, 2))

    @property
    def device(self):
        return self.trans_prob.device

    def build(self, state_types, edges, labels):
        """
        Fills the transition tensor (one-hot per action: trans_prob[a, src, dst] = 1)
        and the accepting/rejecting indicator matrices from the DFA's edges and state types
        """
        labels_map = {label: i for i, label in enumerate(labels + ['end'])}
        with torch.no_grad():
            for (src, dst, label) in edges:
                self.trans_prob[labels_map[label], src, dst] = 1.0
            for s in state_types:
                self.accepting_matrix[s, int(state_types[s] == 1)] = 1.0
                self.rejecting_matrix[s, int(state_types[s] == -1)] = 1.0

    def forward(self, action_seq):
        """
        Runs a batch of one-hot action sequences through the DFA starting
        from state 0, returning the final state distribution and reward
        after the last step
        """
        batch_size, sequence_len, _ = action_seq.shape

        state = torch.zeros(batch_size, self.n_states, device=self.device)
        state[:, 0] = 1.0
        for i in range(sequence_len):
            state = self.step(state, action_seq[:, i])
        reward = state @ self.accepting_matrix
        return state, reward

    def step(self, state, action):
        """
        Soft/differentiable transition: for each batch item, combine the
        current state distribution with the transition tensor to get,
        for the taken action, the distribution over next states
        """
        selected_prob = state.unsqueeze(1).unsqueeze(-2) @ self.trans_prob
        next_state = action.unsqueeze(1) @ selected_prob.squeeze()
        return next_state.squeeze(1)

    def unroll(self, action_seq):
        """
        Like forward, but returns the full sequence of states at every
        timestep instead of only the final one
        """
        batch_size, sequence_len, _ = action_seq.shape
        state = torch.zeros(batch_size, self.n_states, device=self.device)
        state[:, 0] = 1.0

        states = state.new_zeros(batch_size, sequence_len, self.n_states)
        for i in range(sequence_len):
            state = self.step(state, action_seq[:, i])
            states[:, i] = state
        return states

    def next_states_rejecting(self, states):
        """
        For each timestep's state distribution, computes the distribution
        over next states for every possible action in parallel, then
        checks per action whether that resulting state is rejecting
        with probability > 0.5 — used to flag which next actions would
        lead into a dead/rejecting state
        """
        next_dist = torch.einsum('bts,asu->btau', states, self.trans_prob)
        reject_col = self.rejecting_matrix[:, 1]
        reject_prob = torch.einsum('btau,u->bta', next_dist, reject_col)
        return reject_prob > 0.5
