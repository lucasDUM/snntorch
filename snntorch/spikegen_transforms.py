import torch
import numpy as np
from snntorch import spikegen


dtype = torch.float

# Can handle Video data
class DeltaCoding(object):
    def __init__(self, threshold=0.1, padding=False, off_spike=False, alt_order=True):
        self.threshold = threshold
        self.padding = padding
        self.off_spike = off_spike
        self.alt_order = alt_order

    def __call__(self, data):
        return spikegen.delta(data, self.threshold, self.padding, self.off_spike, self.alt_order)

# Can handle Video data
# Can handle Image data
class RateCoding(object):
    def __init__(self, num_steps=False, gain=1, offset=0, first_spike_time=0, time_var_input=False):
        self.num_steps = num_steps
        self.gain = gain
        self.offset = offset
        self.first_spike_time = first_spike_time
        self.time_var_input = time_var_input

    def __call__(self, data):
        return spikegen.rate(data, self.num_steps, self.gain, self.offset, self.first_spike_time, self.time_var_input)