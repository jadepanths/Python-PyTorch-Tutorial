import torch

rank_0_tensor = torch.tensor(4)
print("rank 0")
print(rank_0_tensor)
print(rank_0_tensor.shape)

rank_1_tensor = torch.tensor([1, 2, 3, 4, 5, ])
print("rank 1")
print(rank_1_tensor)
print(rank_1_tensor.shape)

rank_2_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print("rank 2")
print(rank_2_tensor)
print(rank_2_tensor.shape)

rank_3_tensor = torch.tensor(
    [[[0, 1, 2, 3, ],
      [4, 5, 6, 7, ]],
     [[8, 9, 10, 11, ],
      [12, 13, 14, 15, ]],
     [[16, 17, 18, 19, ],
      [20, 21, 22, 23]], ])

print("rank 3")
print(rank_3_tensor)
print(rank_3_tensor.shape)
print()

print("test cuda")
tensor_cpu = torch.rand([2, 2, 2, ])

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor_cuda = tensor_cpu.to('cuda')

print(tensor_cuda)
