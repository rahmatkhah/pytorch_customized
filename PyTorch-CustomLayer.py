
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
""" Custom ReLU function """
class customReLuFn(Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        output = input.clone()
        output[output < 0] = alpha.expand_as(output[output < 0]) * output[output < 0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_variables
        grad_input = grad_alpha = None

        grad_input = alpha.expand_as(grad_output).clone()
        grad_input[input > 0] = 1

        grad_alpha = Variable(torch.zeros(grad_output.size()))
        grad_alpha[input <= 0] = input[input <= 0]
        
        return grad_output * grad_input, (grad_output * grad_alpha).sum()

class customReLU(nn.Module):
    def __init__(self):
        super(customReLU, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.alpha.data.uniform_(0, 0.1)

    def forward(self, input):
        return customReLuFn.apply(input, self.alpha)

#%%
""" Custom Conv2D function """
class customConv2DFn(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        np_input = input.numpy()
        np_weight = weight.numpy()
        n, m = np_input.shape
        p, q = np_weight.shape
        output = torch.from_numpy(
                     np.fft.ifftn(
                        np.multiply(
                            np.fft.fftn(np_weight, (n+p-1, m+q-1)), 
                            np.fft.fftn(np_input), (n+p-1, m+q-1)
                        ), (n, m)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weigth = ctx.saved_variables
        grad_input = grad_weigth = None

        np_input = input.numpy()
        np_weight = weight.numpy()
        np_grad_output = grad_output.numpy()
        n, m = np_input.shape
        p, q = np_weight.shape
        output = Variable(torch.from_numpy(
                     np.fft.ifftn(
                        np.multiply(
                            np.fft.fftn(np_weight, (n+p-1, m+q-1)), 
                            np.fft.fftn(np_input), (n+p-1, m+q-1)
                        ), (n, m))))

        return grad_output * grad_input, grad_weigth

class customConv2D(nn.Module):
    def __init__(self):
        super(customReLU, self).__init__()

    def forward(self, input):
        return customReLuFn.apply(input)
# In[3]:


# Define the CNN architecture (LeNet)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.conv2 = nn.Conv2d(1, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.relu1 = customReLU()
#        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.maxpool2 = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        out = self.relu1(self.conv1(x))
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

