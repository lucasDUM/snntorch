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

class MNIST_SNN_BURST(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size, burst, device):
        super().__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        # Initialize layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

        #self.fc1 = nn.Linear(784, 100)
        print(burst)
        self.fc1 = connections.Linear_Burst(784, 100, burst, device=device)
        #self.fc2 = nn.Linear(100, 10)
        self.fc2 = connections.Linear_Burst(100, 10, burst, device=device)

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step].view(self.batch_size, -1)
            current1 = self.fc1(step, start)
            spk1 = self.lif1(current1)
            current2 = self.fc2(step, spk1)
            spk2, _ = self.lif2(current2)
            spk_rec.append(spk2)
        return torch.stack(spk_rec)

class MNIST_SNN_PHASE(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()
        self.num_steps = num_steps
        self.batch_size = batch_size
        # Initialize layers
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

        #self.fc1 = nn.Linear(784, 100)
        self.fc1 = connections.Linear_Phase(784, 100, num_steps)
        self.fc2 = nn.Linear(100, 10)
        #self.fc2 = connections.Linear_Burst(100, 10, burst, device=device)

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

class VGG_5(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
       
        self.fc1 = nn.Linear(128*4*4, 10)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]
            # BLOCK 1
            current1 = self.conv1(start)
            spk1 = self.lif1(current1)

            # BLOCK 2
            current2 = self.conv2(spk1)
            current2 = F.max_pool2d(current2, 2)
            spk2 = self.lif2(current2)

            # BLOCK 3
            current3 = self.conv3(spk2)
            spk3 = self.lif3(current3)

            # BLOCK 4
            current4 = self.conv4(spk3)
            current4 = F.max_pool2d(current4, 2)
            spk4 = self.lif4(current4)

            # BLOCK 5
            current5 = self.fc1(spk4.view(self.batch_size, -1))
            spk5, _ = self.lif5(current5)

            spk_rec.append(spk5)

        return torch.stack(spk_rec)

class VGG_9(nn.Module):
    def __init__(self, beta, threshold, spike_grad, init_hidden, num_steps, batch_size):
        super().__init__()

        # Initialize layers
        # in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
       
        self.fc1 = nn.Linear(256*2*2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif6 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif7 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif8 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.lif9 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        

        self.num_steps = num_steps
        self.batch_size = batch_size

    def forward(self, x):
        # Record the final layer
        spk_rec = []
        for step in range(self.num_steps):
            start = x[:, step]
            # BLOCK 1
            current1 = self.conv1(start)
            spk1 = self.lif1(current1)

            # BLOCK 2
            current2 = self.conv2(spk1)
            spk2 = self.lif2(current2)

            # BLOCK 3
            current3 = self.conv3(spk2)
            current3 = F.max_pool2d(current3, 2)
            spk3 = self.lif3(current3)

            # BLOCK 4
            current4 = self.conv4(spk3)
            spk4 = self.lif4(current4)

            # BLOCK 5
            current5 = self.conv5(spk4)
            spk5 = self.lif5(current5)

            # BLOCK 6
            current6 = self.conv6(spk5)
            current6 = F.max_pool2d(current6, 2)
            spk6 = self.lif6(current6)

            current7 = self.fc1(spk6.view(self.batch_size, -1))
            spk7 = self.lif7(current7)

            #print(spk7.size())
            current8 = self.fc2(spk7)
            spk8 = self.lif8(current8)

            current9 = self.fc3(spk8)
            spk9, _ = self.lif9(current9)

            spk_rec.append(spk9)

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