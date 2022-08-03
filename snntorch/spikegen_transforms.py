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

# Can handle Image data
class LatencyCoding(object):
    def __init__(self, num_steps=False, threshold=0.01, tau=1, first_spike_time=0, on_target=1, off_target=0, 
                clip=False, normalize=False, linear=False, interpolate=False, bypass=False, epsilon=1e-7):
        self.num_steps = num_steps
        self.threshold = threshold
        self.tau = tau
        self.first_spike_time = first_spike_time
        self.on_target = on_target
        self.off_target = off_target
        self.clip = clip
        self.normalize = normalize
        self.linear = linear
        self.interpolate = interpolate
        self.bypass = bypass
        self.epsilon = epsilon

    def __call__(self, data):
        return spikegen.rate(data, self.num_steps, self.threshold, self.tau, self.first_spike_time, self.on_target, self.off_target, 
                            self.clip, self.normalize, self.linear, self.interpolate, self.bypass, self.epsilon)

