import torch

# dot multiplication
t1 = torch.randn(1, 2, )
t2 = torch.randn(1, 2, )

tMul = torch.mul(t1, t2)

print("t1: \n", t1)
print("t2: \n", t2, "\n")

print("dot multiplication: \n", tMul, "\n")

print("another example: ")
t1 = torch.tensor([3, 2])
t2 = torch.tensor([[5, 6],
                  [7, 8],
                   [1, 6]])

tMul = torch.mul(t1, t2)

print("t1: \n", t1)
print("t2: \n", t2, "\n")
print("dot multiplication: \n", tMul, "\n"
      , tMul.shape)

##########################################################
# Matrix Multiplication

print("\n Matrix Multiplication")
t1 = torch.tensor([[1, 2, 3, 4, ],
                   [1, 2, 3, 4, ],
                   [1, 2, 3, 4, ]])

print("t1: \n", t1, "\n", t1.shape)

t2 = torch.tensor([[1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2]])

print("t2: \n", t2, "\n", t2.shape)

tMM = torch.mm(t1, t2)
print("matrix multiplication: \n", tMM, "\n"
      , tMM.shape)


# torch.matmul(a, b)
t1 = torch.ones(2, 4, 2)
t2 = torch.ones(2, 2, 3)
tMatmul = torch.matmul(t1, t2)
print("matrix multiplication: \n", tMatmul, "\n"
      , tMatmul.shape)

