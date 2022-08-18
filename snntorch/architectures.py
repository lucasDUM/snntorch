#######################################################################
################################ MNIST ################################
#######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import connections


class MNIST_SNN(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        # Initialize layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step].view(self.batch_size, -1)
            current1 = self.fc1(start)
            spk1 = self.lif1(current1)
            current2 = self.fc2(spk1)
            spk2 = self.lif2(current2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec)
class MIST_CNN_SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv_burst_1 = nn.Conv2d(1, 12, 5)
        self.conv_burst_2 = nn.Conv2d(12, 64, 5)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.linear_burst_1 = nn.Linear(64*4*4, 10)

    def forward(self, x, num_steps):
        # Record the final layer
        spk_rec = []
        for step in range(num_steps):
            start = x[:, step]

            current1, threshold = self.conv_burst_1(start)
            current1 = F.max_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2, threshold = self.conv_burst_2(spk1)
            current2 = F.max_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3, threshold = self.linear_burst_1(spk2.view(batch_size, -1))
            spk3 = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

class MIST_CNN_SNN_2(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.fc1 = nn.Linear(64*4*4, 10)

    def forward(self, x, num_steps):
        # Record the final layer
        spk_rec = []
        for step in range(num_steps):
            start = x[:, step]

            current1 = self.conv1(start)
            current1 = F.avg_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2 = self.conv2(spk1)
            current2 = F.avg_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3 = self.fc1(spk2.view(batch_size, -1))
            spk3 = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

#CNN: Input-32C3-AP2-32C3-AP2-128FC-10


class Burst_MNIST_CNN_SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv_burst_1 = connections.Conv2d_Burst(1, 12, 5, 2)
        self.conv_burst_2 = connections.Conv2d_Burst(12, 64, 5, 2)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.linear_burst_1 = connections.Linear_Burst(64*4*4, 10, 2)

    def forward(self, x, num_steps):
        # Record the final layer
        spk3_rec = []
        for step in range(num_steps):
            start = x[:, step]

            current1, threshold = self.conv_burst_1(start)
            current1 = F.max_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2, threshold = self.conv_burst_2(spk1)
            current2 = F.max_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3, threshold = self.linear_burst_1(spk2.view(batch_size, -1))
            spk3 = self.lif3(current3)
            spk3_rec.append(spk3)

        return torch.stack(spk3_rec)



# Load the network onto CUDA if available
#burst_net = Burst_SNN().to(device)

# Rethinking the performance comparison between SNNS and ANNS
# https://www.sciencedirect.com/science/article/abs/pii/S0893608019302667

# MNIST 
#CNN: Input-32C3-AP2-32C3-AP2-128FC-10
# CIFAR10 
#CNN: Input-64C3-AP2-128C3-128C3-AP2-256FC-10
# N-MNIST

# DVS-CIFAR10
#CNN: Input-64C3-AP2-128C3-AP2-256FC-10

# Going Deeper With Directly-Trained Larger Spiking Neural Networks
# file:///D:/Downloads/17320-Article%20Text-20814-1-2-20210518.pdf

# DVS-gesture

# CIFAR10 

# Rate Coding Or Direct Coding: Which One Is Better For Accurate, Robust, And Energy-Efficient Spiking Neural Networks?
# https://ieeexplore.ieee.org/abstract/document/9747906?casa_token=hmZrkHBJoXkAAAAA:q8f8ZLvedovJk1M2ddItAwN9_NtNjiPBQ9-PnAKSASIuyS75C8gbMyYvBgiK6kKDMpDy-ordBQ
# VGG5, VGG9

#Direct Training for Spiking Neural Networks: Faster, Larger, Better
#https://ojs.aaai.org/index.php/AAAI/article/view/3929

# populations encoding

# CIFAR10
#SMall 128C3(Encoding)-AP2-256C3-AP2-256FC-Voting
#Middle 128C3(Encoding)-AP2-256C3-512C3-AP2-512FC-Voting
#Large 128C3(Encoding)-256C3-AP2-512C3-AP2-1024C3-512C3-1024FC-512FC-Voting
# 128C3(Encoding)-256C3-AP2-512C3-AP2-1024C3-512C3-1024FC-512FC-Voting

# Temporal-Coded Deep Spiking Neural Network with Easy Training and Robust Performance
#https://github.com/zbs881314/Temporal-Coded-Deep-SNN

# SCNN(5,32,2) → SCNN(5,16,2) → FC(10)
#CIFAR-10 SpikingVGG16: SCNN(3,64,1) → SCNN(3,64,1) → MP(2) → SCNN(3,128,1) → SCNN(3,128,1)
#→ MP(2) → SCNN(3,256,1) → SCNN(3,256,1) → SCNN(3,256,1) → MP(2) → SCNN(3,512,1)
#→ SCNN(3,512,1) → SCNN(3,512,1) → MP(2) → SCNN(3,1024,1) → SCNN(3,1024,1)
#→ SCNN(3,1024,1) → MP(2) → FC(4096) → FC(4096) → FC(512) → FC(10)
#

# Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch
# VGG9

# The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks
#mnist 
#784 
#100 
#10