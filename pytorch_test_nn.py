import torch
import torch.nn as nn

# create feed forward layer of 2 inputs and 5 outputs
l = nn.Linear(2, 5)

# create 2 input tensor
v = torch.FloatTensor([1, 2])

# print 5 output tensor
print(l(v))

# create 3 layer NN with softmax output applied to along dimension 1 (minibatch)(dimension 0 is batch samples),
# ReLU nonlinears and dropout
s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)

print(s)
print(s(torch.FloatTensor([[1, 2]])))


