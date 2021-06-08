import torch

t1 = torch.tensor([[1, 2],
                   [3, 4]])

t2 = torch.tensor([[5, 6],
                   [7, 8]])

tStack = torch.stack((t1, t2))
print("stack: \n", tStack)
print("stack dimension: ", tStack.shape)
print()

tCatDim1 = torch.cat((t1, t2), dim=0)
print("cat | dim=0: \n", tCatDim1)
print("cat | new dimension: ", tCatDim1.shape)
print()

tCatDim2 = torch.cat((t1, t2), dim=1)
print("cat | dim=1: \n", tCatDim2)
print("cat | new dimension: ", tCatDim2.shape)