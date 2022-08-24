import torch
import numpy as np
from snntorch import spikegen
from snntorch import connections

dtype = torch.float

# Can handle Image data
class hybrid_encoding_image(object):
    def __init__(self, separate=True, splits = [0.5, 0.5], encodings=["latency", "rate"], 
                num_steps=False, time_var_input=False, tau=1, threshold=0.01, clip=False, linear=False, interpolate=False, gain=1, N_max=5, T_min=2):
        self.separate = separate
        self.splits = splits
        self.encodings = encodings
        self.num_steps = num_steps
        self.time_var_input = time_var_input
        self.tau = tau
        self.threshold = threshold
        self.clip = clip
        self.linear = linear
        self.interpolate = interpolate
        self.gain = gain
        self.N_max = N_max
        self.T_min = T_min

    def __call__(self, data):
        return spikegen.hybrid_encoding_image(data, self.separate, self.splits, self.encodings, self.num_steps, self.time_var_input, 
                                            self.tau, self.threshold, self.clip, self.linear, self.interpolate, self.gain, self.N_max, self.T_min)


# Can handle Video data
class SaccadeCoding(object):
    def __init__(self, timesteps=100, max_dx=20, max_dy=20, delta_threshold=0.1):
        self.timesteps = timesteps
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.delta_threshold = delta_threshold

    def __call__(self, data):
        return spikegen.saccade_coding(data, self.timesteps, self.max_dx, self.max_dy, self.delta_threshold)

# Can handle Video data
class DeltaCoding(object):
    def __init__(self, threshold = 0.1, padding=False, off_spike=False, alt_order=True):
        self.threshold = threshold
        self.padding = padding
        self.off_spike = off_spike
        self.alt_order = alt_order

    def __call__(self, data):
        return spikegen.delta(data, self.threshold, self.padding, self.off_spike, self.alt_order)

# Can handle Image data
class PhaseCoding(object):
    def __init__(self, timesteps, is_weighted=False):
        self.timesteps = timesteps
        self.is_weighted = is_weighted

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return spikegen.phase_coding(images, self.timesteps, self.is_weighted)

# Can handle Image data
class BurstCoding(object):
    def __init__(self, N_max = 5, timesteps = 100, T_min = 2):
        self.N_max = N_max
        self.timesteps = timesteps
        self.T_min = T_min

    def __call__(self, data):
        return spikegen.burst_coding(data, self.N_max, self.timesteps, self.T_min)

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
        return spikegen.latency(data, self.num_steps, self.threshold, self.tau, self.first_spike_time, self.on_target, self.off_target, 
                            self.clip, self.normalize, self.linear, self.interpolate, self.bypass, self.epsilon)

