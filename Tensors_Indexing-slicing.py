import torch

tensor = torch.rand(4, 4)
print(f'Initial Tensor: \n {tensor} \n')

print('Rows:')
print('First row: ', tensor[0])
print('Third row: ', tensor[2])
print('Last row:  ', tensor[-1], '\n')

print('Column')
print('First column (using ":"): \n', tensor[:, 0])
print('First Column (using "..."): \n', tensor[..., 0])
print('Second Column: \n', tensor[:, 1])
print('Last column: \n', tensor[..., -1])
print('Second to last column \n', tensor[:, -2])

tensor[:, 1] = 0
print('tensor after slicing \n', tensor)
