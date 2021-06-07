import torch
import numpy as np

data = (2, 3)
x_data = torch.rand(data)
print(f"x_data from arrays: \n {x_data} \n")

data2 = [[1, 2], [3, 4]]
np_array = np.array(data2)
tensor_from_np = torch.from_numpy(np_array)
print(f"From Numpy: \n {tensor_from_np} \n")

shape = (4, 3, 2)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

tensor = torch.rand(3, 4, 2)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device} \n")

