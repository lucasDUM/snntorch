import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import connections

#######################################################################
################################ MNIST ################################
#######################################################################

# The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks
# 1
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
            spk2, _ = self.lif2(current2)
            spk_rec.append(spk2)
        return torch.stack(spk_rec)

# snnTorch
# 2
class MIST_CNN_SNN(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv_burst_1 = nn.Conv2d(1, 12, 5)
        self.conv_burst_2 = nn.Conv2d(12, 64, 5)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.linear_burst_1 = nn.Linear(64*4*4, 10)

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]

            current1 = self.conv_burst_1(start)
            current1 = F.max_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2 = self.conv_burst_2(spk1)
            current2 = F.max_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3 = self.linear_burst_1(spk2.view(self.batch_size, -1))
            spk3, _ = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

# Rethinking the performance comparison between SNNS and ANNS
# 3
class MIST_CNN_SNN_2(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.fc1 = nn.Linear(32*4*4, 10)

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]

            current1 = self.conv1(start)
            current1 = F.avg_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2 = self.conv2(spk1)
            current2 = F.avg_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3 = self.fc1(spk2.view(self.batch_size, -1))
            spk3, _ = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)


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


# SIMLE MODEL
# CNN 1: Input-32C3-AP2-32C3-AP2-128FC-10

# Revisiting Batch Normalization for Training Low-Latency Deep Spiking Neural Networks From Scratch
# VGG9

# Rethinking the performance comparison between SNNS and ANNS
class CIFAR_CNN_SNN(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, burst param
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
        self.fc1 = nn.Linear(32*4*4, 10)

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]

            current1 = self.conv1(start)
            current1 = F.avg_pool2d(current1, 2)
            spk1 = self.lif1(current1)
            current2 = self.conv2(spk1)
            current2 = F.avg_pool2d(current2, 2)
            spk2 = self.lif2(current2)
            current3 = self.fc1(spk2.view(self.batch_size, -1))
            spk3, _ = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)

# Temporal-Coded Deep Spiking Neural Network with Easy Training and Robust Performance
# CIFAR-10 SpikingVGG16: SCNN(3,64,1) → SCNN(3,64,1) → 

#MP(2) → SCNN(3,128,1) → SCNN(3,128,1)

#→ MP(2) → SCNN(3,256,1) → SCNN(3,256,1) → SCNN(3,256,1) → MP(2) → SCNN(3,512,1)

#→ SCNN(3,512,1) → SCNN(3,512,1) → MP(2) → SCNN(3,1024,1) → SCNN(3,1024,1)
#→ SCNN(3,1024,1) → MP(2) → FC(4096) → FC(4096) → FC(512) → FC(10)
class VGG_16(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        # MAX POOL
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        # MAX POOL
        self.conv5 = nn.Conv2d(128, 256, 3, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1)
        # MAX POOL
        self.conv8 = nn.Conv2d(256, 512, 3, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1)
        # MAX POOL
        self.conv11 = nn.Conv2d(512, 1024, 3, 1)
        self.conv12 = nn.Conv2d(1024, 1024, 3, 1)
        self.conv13 = nn.Conv2d(1024, 1024, 3, 1)
        # MAX POOL
        self.fc1 = nn.Linear(1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 512)
        self.fc4 = nn.Linear(512, 10)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif8 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif9 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif10 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif11 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif12 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif13 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif14 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif15 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif16 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]

            # BLOCK 1
            current1 = self.conv1(start)
            current1 = F.max_pool2d(current1, 2)
            spk1 = self.lif1(current1)

            # BLOCK 2
            current2 = self.conv2(spk1)
            current2 = F.max_pool2d(current2, 2)
            spk2 = self.lif2(current2)

            # BLOCK 3
            current3 = self.conv3(spk2)
            current3 = F.max_pool2d(current3, 2)
            spk3 = self.lif3(current3)

            # BLOCK 4
            current4 = self.conv4(spk3)
            current4 = F.max_pool2d(current4, 2)
            spk4 = self.lif4(current4)

            # BLOCK 5
            current5 = self.conv5(spk4)
            current5 = F.max_pool2d(current5, 2)
            spk5 = self.lif5(current5)

            # BLOCK 6
            current6 = self.conv6(spk5)
            current6 = F.max_pool2d(current6, 2)
            spk6 = self.lif6(current6)

            # BLOCK 7
            current7 = self.conv7(spk6)
            current7 = F.max_pool2d(current7, 2)
            spk7 = self.lif7(current7)

            # BLOCK 8
            current8 = self.conv8(spk7)
            current8 = F.max_pool2d(current8, 2)
            spk8 = self.lif8(current8)

            # BLOCK 9
            current9 = self.conv9(spk8)
            current9 = F.max_pool2d(current9, 2)
            spk9 = self.lif9(current9)

            # BLOCK 10
            current10 = self.conv10(spk9)
            current10 = F.max_pool2d(current10, 2)
            spk10 = self.lif10(current10)

            # BLOCK 11
            current11 = self.conv11(spk10)
            current11 = F.max_pool2d(current11, 2)
            spk11 = self.lif11(current11)

            # BLOCK 12
            current12 = self.conv12(spk11)
            current12 = F.max_pool2d(current12, 2)
            spk12 = self.lif12(current12)

            # BLOCK 13
            current13 = self.conv8(spk12)
            current13 = F.max_pool2d(current13, 2)
            spk13 = self.lif9(current13)

            print(spk13.size())
            raise Exception("PLS STOP HERE")
            # BLOCK 14
            # BLOCK 15
            # BLOCK 16


            current3 = self.fc1(spk2.view(self.batch_size, -1))
            spk3, _ = self.lif3(current3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)


# Load the network onto CUDA if available
#burst_net = Burst_SNN().to(device)

# Rethinking the performance comparison between SNNS and ANNS
# https://www.sciencedirect.com/science/article/abs/pii/S0893608019302667

# MNIST 
#CNN: Input-32C3-AP2-32C3-AP2-128FC-10
# CIFAR10 
#CNN: 
# N-MNISTInput-64C3-AP2-128C3-128C3-AP2-256FC-10

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
# CIFAR-10 SpikingVGG16: SCNN(3,64,1) → SCNN(3,64,1) → MP(2) → SCNN(3,128,1) → SCNN(3,128,1)
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