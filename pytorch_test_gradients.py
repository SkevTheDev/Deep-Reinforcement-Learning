import torch

v1 = torch.tensor([1.0, 1.0], requires_grad=True)
v2 = torch.tensor([2.0, 2.0])
v_sum = v1 + v2
v_res = (v_sum*2).sum()
print(v_res)

# is_leaf returns true if tensor was constructed by user or false if it is the result of function
print(v1.is_leaf)
print(v2.is_leaf)

print(v_sum.is_leaf)
print(v_res.is_leaf)

# requires_grad by default is set to false for is_leaf=True.
print(v1.requires_grad)
print(v2.requires_grad)
print(v_sum.requires_grad)
print(v_res.requires_grad)

# calculate gradients
v_res.backward()
print(v1.grad)
