import torch
import numpy as np
from snntorch import spikegen


dtype = torch.float

class DeltaCoding(object):
    def __init__(self, threshold=0.1, padding=False, off_spike=False, alt=True):
        self.threshold = threshold
        self.padding = padding
        self.off_spike = off_spike
        self.alt = alt

    def __call__(self, data):
        print(self.threshold)
        return delta(data, self.threshold, self.padding, self.off_spike, self.alt)