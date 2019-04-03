import torch
import numpy as np

# create a tensor
a = torch.FloatTensor(3, 2)

print(a)

# functional operation that zeroizes a tensor by creating a copy and leaving the original alone
# zero() is not a torch function
# b = a.zero()
# print(b)
# print(a)

# inplace operation that zeroizes the a tensor (computationally more efficient)
# underscore defines inplace
a.zero_()
print(a)

# create a tensor from a python list
b = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(b)

# same object created in numpy
n = np.zeros(shape=(3, 2))
print(n)

# defaults to creating 64-bit tensor
c = torch.tensor(n)
print(c)

# creating 32 bit is more computationally efficient
n2 = np.zeros(shape=(3, 2), dtype=np.float32)
d = torch.tensor(n2)
print(d)

e = torch.tensor(n, dtype=torch.float32)
print(e)


# pytorch supports zero dimension tensors
a = torch.tensor([1, 2, 3])
print(a)

s = a.sum()
print(s)
print(s.item())
print(torch.tensor(1))
