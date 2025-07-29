from datetime import datetime
import math
import statistics
import numpy as np
import torch
import pm4py
from pm4py.objects.log.obj import EventLog


class Log:
    def __init__(self, root_path, dataset, portion):
        self.folder_path = f'{root_path}datasets/{dataset}/log/'
        self.portion = portion
        self.event_log = pm4py.convert_to_event_log(pm4py.read_xes(f'{self.folder_path}{self.portion}.xes'))
        self.event_names = []

    def order(self):
        def get_trace_date(trace):
            date = trace[0].get('time:timestamp')
            # date = trace.attributes['time:timestamp']
            return date if isinstance(date, datetime) else datetime.max

        log_sorted = sorted(self.event_log, key=get_trace_date)
        self.event_log = log_sorted
        pm4py.write_xes(EventLog(log_sorted), f'{self.folder_path}ordered.xes')

    def split_train_test(self):
        split_index = int(len(self.event_log) * 0.8)
        train_log = EventLog(self.event_log[:split_index])
        test_log = EventLog(self.event_log[split_index:])

        pm4py.write_xes(train_log, f'{self.folder_path}train_80.xes')
        pm4py.write_xes(test_log, f'{self.folder_path}test_20.xes')

    def get_event_names(self):
        event_names = []
        for trace in self.event_log:
            for event in trace:
                event_name = f"a_{(event['concept:name']).lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(',  '_').replace(')', '_')}"
                if event_name not in event_names:
                    event_names.append(event_name)
        self.event_names = event_names
        return event_names

    def encode(self, event_names):
        self.event_names = event_names

        event_to_idx = {event: i for i, event in enumerate(self.event_names)}
        end_idx = len(self.event_names)
        num_classes = len(self.event_names) + 1
        max_trace_len = max(len(trace) for trace in self.event_log) + 1

        end_vec = np.zeros(num_classes, dtype=int)
        end_vec[end_idx] = 1
        encoded_traces = []
        for trace in self.event_log:
            encoded_trace = []

            for event in trace:
                vec = np.zeros(num_classes, dtype=int)
                event_name = f"a_{(event['concept:name']).lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '_').replace(')', '_')}"
                vec[event_to_idx[event_name]] = 1
                encoded_trace.append(vec)
            while len(encoded_trace) < max_trace_len:
                encoded_trace.append(end_vec.copy())

            trace_flat = ' '.join(str(x) for vec in encoded_trace for x in vec)
            encoded_traces.append(trace_flat)
        with open(f'{self.folder_path}{self.portion}_encoded.txt', 'w') as f:
            f.write('\n'.join(encoded_traces))

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

    def to_tensor(self, portion):
        tensor = torch.tensor(np.loadtxt(f'{self.folder_path}{self.portion}_encoded.txt'))  # , max_rows=10000)) # pylint: disable=no-member
        if portion != 1:
            indices = torch.randperm(tensor.size(0))[:int(tensor.size(0) * portion)]
            tensor = tensor[indices.sort().values]
        tensor = tensor.view(tensor.size(0), -1, len(self.event_names) + 1)
        return tensor.float()
