import math
from datetime import datetime
from copy import deepcopy
import statistics
import random
import numpy as np
import torch
import pm4py
from pm4py.objects.log.obj import EventLog
from common.declare_model import clean_activity_name


class Log:
    def __init__(self, folder_path, filename):
        self.folder_path = folder_path
        self.filename = filename
        self.event_log = pm4py.convert_to_event_log(pm4py.read_xes(str(folder_path / f'{self.filename}.xes')))
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

    def encode(self, event_names):
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

    def order(self):
        def get_trace_date(trace):
            date = trace[0].get('time:timestamp')
            # date = trace.attributes['time:timestamp']
            return date if isinstance(date, datetime) else datetime.max

        log_sorted = EventLog(sorted(self.event_log, key=get_trace_date))
        pm4py.write_xes(self.event_log, self.folder_path / 'ordered.xes')
        self.event_log = log_sorted

    def split_train_test(self):
        split_index = int(len(self.event_log) * 0.8)
        train_log = EventLog(self.event_log[:split_index])
        test_log = EventLog(self.event_log[split_index:])

        pm4py.write_xes(train_log, self.folder_path / 'train_80.xes')
        pm4py.write_xes(test_log, self.folder_path / 'test_20.xes')

    def add_noise(self, noise_level):
        noised_log = deepcopy(self.event_log)

        activity_names = set()
        for trace in noised_log:
            for event in trace:
                activity_names.add(event['concept:name'])
        activity_names = list(activity_names)

        noise_perc = noise_level / 100
        num_to_substitute = int(sum(len(trace) for trace in noised_log) * noise_perc)

        all_events = [(i, j) for i, trace in enumerate(noised_log) for j, _ in enumerate(trace)]
        to_substitute_indices = set(random.sample(all_events, num_to_substitute))

        for i, j in to_substitute_indices:
            current_name = noised_log[i][j]['concept:name']
            choices = [name for name in activity_names if name != current_name]
            if choices:
                new_name = random.choice(choices)
                noised_log[i][j]['concept:name'] = new_name

        pm4py.write_xes(noised_log, self.folder_path / f'train_80_n{noise_level}.xes')

    def get_first_prefix(self):
        traces_lengths = [len(trace) for trace in self.event_log]
        median = statistics.median(traces_lengths)
        return math.floor(median / 2)