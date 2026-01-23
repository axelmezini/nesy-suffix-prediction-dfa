import os
import math
import statistics
import numpy as np
import torch
import pm4py
from common.declare_model import clean_activity_name


#TODO: log noise injection da implementare
class Log:
    def __init__(self, root_path, dataset, filename):
        folder_path = str(os.path.join(root_path, 'datasets', dataset, 'log'))
        self.filename = filename
        self.event_log = pm4py.convert_to_event_log(pm4py.read_xes(os.path.join(folder_path, f'{self.filename}.xes')))
        self.event_names = []
        self.tensor = None

    def define_event_names(self):
        event_names = []
        for trace in self.event_log:
            for event in trace:
                event_name = clean_activity_name(event['concept:name'])
                if event_name not in event_names:
                    event_names.append(event_name)
        return event_names

    def encode(self, event_names, subset=1):
        self.event_names = event_names
        event_to_idx = {event: i for i, event in enumerate(self.event_names)}
        num_classes = len(self.event_names) + 1
        max_trace_len = max(len(trace) for trace in self.event_log) + 1

        end_vec = np.zeros(num_classes, dtype=int)
        end_vec[len(self.event_names)] = 1

        encoded_traces = []
        for trace in self.event_log:
            encoded_trace = []
            for event in trace:
                vec = np.zeros(num_classes, dtype=int)
                event_name = clean_activity_name(event['concept:name'])
                vec[event_to_idx[event_name]] = 1
                encoded_trace.append(vec)

            while len(encoded_trace) < max_trace_len:
                encoded_trace.append(end_vec.copy())
            encoded_traces.append(encoded_trace)

        encoded_np = np.asarray(encoded_traces, dtype=np.float32)
        self.tensor = torch.from_numpy(encoded_np)
        return self.tensor

    def decode(self, encoded_traces):
        traces_strings = []

        for i in range(encoded_traces.size(0)):
            trace_events = []

            numpy_array = encoded_traces[i].cpu().numpy()
            for event in numpy_array:
                idx = event.argmax()
                if idx < len(self.event_names):
                    trace_events.append(f'{self.event_names[idx]}')
                elif idx == len(self.event_names):
                    trace_events.append('end')
                    break
            traces_strings.append(', '.join(trace_events))

        return '\n'.join(traces_strings)

    def get_first_prefix(self):
        traces_lengths = [len(trace) for trace in self.event_log]
        median = statistics.median(traces_lengths)
        return math.floor(median / 2)

    def get_subset(self, portion):
        if 0 < portion < 1:
            N = self.tensor.size(0)
            k = max(1, int(round(N * portion)))
            idx = torch.randperm(N)[:k].sort().values
            return self.tensor.index_select(0, idx)
        else:
            return self.tensor