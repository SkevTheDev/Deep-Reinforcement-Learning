import torch

a = torch.FloatTensor([2, 3])
print(a)

ca = a.cuda()
print(ca)

print(a + 1)
print(ca + 1)

print(ca.device)
