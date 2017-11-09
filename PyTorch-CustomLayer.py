
# coding: utf-8

# In[1]:


# CNN Example on MNIST

# Import PyTorch Utilities
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, Function

import numpy as np
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython().magic('matplotlib inline')
except(Exception):
    pass

cuda = False

# In[2]:


# PyTorch supports loading many commonly used datasets

batch_size = 128

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,
                   transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../', train=False, download=True,
                    transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)


#%%
""" Custom ReLU function , 1 input channel, 1 output channel """
class customReLuFn(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        output = input.clone()
        output[output < 0] = alpha.expand_as(
                output[output < 0]) * output[output < 0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_variables
        grad_input = grad_alpha = None

        grad_input = alpha.expand_as(grad_output).clone()
        grad_input[input > 0] = 1

        grad_alpha = Variable(torch.zeros(grad_output.size()))
        if cuda:
            grad_alpha = grad_alpha.cuda()
        grad_alpha[input <= 0] = input[input <= 0]
        
        return grad_output * grad_input, (grad_output * grad_alpha).sum() \
                                          / grad_output.size(0)

class customReLU(nn.Module):
    def __init__(self):
        super(customReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(128,1,1,1))
        self.alpha.data.uniform_(0, 0.1)

    def forward(self, input):
        return customReLuFn.apply(input, self.alpha)

#%%
""" Custom Conv2D function """
class customConv2DFn(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        if cuda:
            input = input.cpu()
            weight = weight.cpu()

        np_input = input.numpy()
        np_weight = weight.numpy()
        p, q = np_weight.shape
        a, b, c, d = np_input.shape
        np_output = np.zeros((a, b, c+p-1, d+q-1))
        for n in range(a):
            for m in range(b):
#               np_output[n,m] = sig.convolve2d(np_input[n,m], np_weight)
                np_output[n,m] = np.fft.ifft2(np.multiply(
                        np.fft.fftn(np_input[n,m], (c+p-1, d+q-1)),
                        np.fft.fftn(np_weight, (c+p-1, d+q-1))
                            )).real
        output = torch.from_numpy(np_output).float()
        
        if cuda:
            output = output.cuda()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weigth = None

        if cuda:
            input = input.cpu()
            weight = weight.cpu()
            grad_output = grad_output.cpu()
        
        np_input = input.numpy()
        np_weight = weight.numpy()
        np_grad_output = grad_output.data.numpy()
        a, b, c, d = np_input.shape
        p, q = np_weight.shape
        np_grad_input = np.zeros((a, b, c, d))
        np_grad_weight = np.zeros((a, b, p, q))
        for n in range(a):
            for m in range(b):
#               np_grad_input[n,m] = sig.convolve2d(
#                       np_grad_output[n,m], 
#                       np_weight[::-1,::-1])[p-1:-(p-1),q-1:-(q-1)]
#               np_grad_weight[n,m] = sig.convolve2d(
#                       np_input[n,m,::-1,::-1], 
#                       np_grad_output[n,m])[c-1:-(c-1),d-1:-(d-1)]
                
                np_grad_input[n,m] = np.fft.ifft2(np.multiply(
                        np.fft.fftn(np_grad_output[n,m], (c+2*p-2, d+2*q-2)), 
                        np.fft.fftn(np_weight[::-1,::-1], (c+2*p-2, d+2*q-2))
                            ))[p-1:-(p-1),q-1:-(q-1)].real
                np_grad_weight[n,m] = np.fft.ifft2(np.multiply(
                        np.fft.fftn(np_input[n,m,::-1,::-1], (2*c+p-2, 2*d+q-2)),
                        np.fft.fftn(np_grad_output[n,m], (2*c+p-2, 2*d+q-2))
                            ))[c-1:-(c-1),d-1:-(d-1)].real

        grad_input = Variable(torch.from_numpy(np_grad_input)).float()
        grad_weight = Variable(torch.from_numpy(
                               np_grad_weight.sum(0).squeeze())).float()
        
        if cuda:
            grad_input = grad_input.cuda()
            grad_weight = grad_weight.cuda()
        return grad_input, grad_weight

class customConv2D(nn.Module):
    def __init__(self):
        super(customConv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(5,5))
#        self.weight.data.randn(0, 0.1)

    def forward(self, input):
        return customConv2DFn.apply(input, self.weight)
# In[3]:


# Define the CNN architecture (LeNet)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = customConv2D()
        self.conv11 = nn.Conv2d(1, 1, 5)
        self.conv12 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu0 = customReLU()
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        out = self.relu1(self.conv0(x))
        out = self.relu1(self.conv11(out))
        out = self.relu1(self.conv12(out))
        out = self.maxpool1(out)
        out = self.relu2(self.conv2(out))
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

# In[4]:


# Initialize Model

ninput = 784
noutput = 10
nweights = 32
dropout = 0.5
model = CNN()
criterion = nn.CrossEntropyLoss()

# if using cuda
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# setup optimization routine
learning_rate = 1e-3
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# In[5]:


# Training Function

log_interval = 200
def train(epoch):
    
    model.train() # Needed for dropout, batch normalization etc.
    
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)
        
        # make sure gradients are reset to zero.        
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1)[1]#, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    
    print('\nTrain Epoch: {} \tAccuracy: {:.6f}'.format(
                epoch, 100. * correct / len(train_loader.dataset)))


# In[6]:


# Testing Function

def test(epoch):
    
    model.eval() # Needed for dropout, batch normalization etc.
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
            target = target.cuda()
        data, target = Variable(data), Variable(target)
        
        # make sure gradients are reset to zero.        
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1)[1]#, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    print('Test Epoch: {} \tAccuracy: {:.6f}\n'.format(
                epoch, 100. * correct / len(test_loader.dataset)))


# In[7]:


# Actual Loop

nepochs = 1000

for epoch in range(nepochs):
    train(epoch)
    test(epoch)

