import torch
import numpy as np
from torch import nn

from typing import Dict, Iterable, Callable, Any

def spike_rate(count, size, time_steps):
	return count / (size*time_steps)

def average_sparsity_per_timestep_per_image(monitor, batch, time_steps, device, num_hidden):
    hidden = torch.tensor([]).to(device)
    output = torch.tensor([]).to(device)
    count = 0
    count1 = 0
    avg_hidden = []
    avg_output = []
    for tensor in monitor.records:
        if str(type(tensor)) == "<class 'torch.Tensor'>":
            # Hidden layers
            if count == 0:
                hidden = tensor
            else:
                hidden = hidden + tensor
            count += 1
            if count == time_steps*num_hidden:
                value = (((hidden != 0.).sum(dim=1)/tensor.size()[1]).sum()/batch).item()
                avg_hidden.append(value)
                count = 0                      
        else:
            # Output layer
            if count1 == 0:
                output = tensor[0]
            else:
                output = output + tensor[0]
            count1 += 1
            if count1 == time_steps:
                value = (((output != 0.).sum(dim=1)/tensor[0].size()[1]).sum()/batch).item()
                avg_output.append(value)
                count1 = 0

    return sum(avg_hidden)/len(avg_hidden), sum(avg_output)/len(avg_output)

def average_spike_per_image(spike_count, total_images):
    return spike_count/total_images

def total_spike_count(monitor):
    total_spike_count = 0
    for tensor in monitor.records:
        if str(type(tensor)) == "<class 'torch.Tensor'>":
            total_spike_count += tensor.sum()
        else:
            total_spike_count += tensor[0].sum()
    return total_spike_count.item()

def total_neurons(monitor):
  sum = 0
  for i in range(len(monitor.monitored_layers)):
    temp = 1
    try:
      for value in monitor.records[i].size()[1:]:
        temp *= value
      sum += temp
    except:
      for value in monitor.records[i][0].size()[1:]:
        temp *= value
      sum += temp
  return sum

def unpack_len1_tuple(x: tuple or torch.Tensor):
    if isinstance(x, tuple) and x.__len__() == 1:
        return x[0]
    else:
        return x

class BaseMonitor:
    def __init__(self):
    	self.hooks = []
    	self.monitored_layers = []
    	self.records = []
    	self.name_records_index = {}
    	self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class OutputMonitor(BaseMonitor):
    def __init__(self, net: nn.Module, instance: Any or tuple = None, function_on_output: Callable = lambda x: x):
        """
        .. _OutputMonitor-en:
        :param net: a network
        :type net: nn.Module
        :param instance: the instance of modules to be monitored. If ``None``, it will be regarded as ``type(net)``
        :type instance: Any or tuple
        :param function_on_output: the function that applies on the monitored modules' outputs
        :type function_on_output: Callable
        """
        super().__init__()
        self.function_on_output = function_on_output

        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.function_on_output(unpack_len1_tuple(y)))
        return hook