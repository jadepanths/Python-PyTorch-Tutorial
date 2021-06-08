# Python-PyTorch-Tutorial
Basic PyTorch Tutorials in Python for beginners

# Work In Progress

# Installation
Install an IDE on your local device like [IDLE](https://www.python.org/downloads/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), or [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/features/python/).  I personally recommend [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) as it has an excellent debugger, sophisticated autocompletion, and refactoring support.

Install [Anaconda](https://www.anaconda.com/products/individual). Anaconda is a toolkit that includes open-source packages and libraries. You will need this to install libraries and manage environments where you will compile your code in.

Check if your computer/labtop is ***CUDA-capable*** CUDA is limited to ***NVIDIA*** GPU. If you know what GPU you have, you can simply check if it CUDA capable here [CUDA GPUS LIST](https://developer.nvidia.com/cuda-gpus). If you do not know what GPU do you have, Open "Run" in the window search bar or the start menu. Then run this command: ```control /name Microsoft.DeviceManager```. Then go to your Display Adapters to check your GPUs.
![Device Manager](https://user-images.githubusercontent.com/85147048/120597340-1eacff00-c46f-11eb-824c-7fcfffb5f5ee.png)
After making sure you have a CUDA-capable GPU, you can download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Install PyTorch
First, create a new environment by openning Anaconda Prompt in the window start menu. Then, run this command: ```conda create --name deeplearning``` where "deeplearning" is the name of the environment and can use any name you wish. The terminal will ask you to proceed, type 'y' and enter. Then, run the following command ```conda activate deeplearning``` (note: "deeplearning" is the environment name you created. Makde sure to use the same name).<br/>
For **NON-CUDA**, run this command <br/>```conda install pytorch torchvision torchaudio cpuonly -c pytorch```.<br/>
For **CUDA-capable**, run this command <br/>```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```.<br/>
Then run 'y' to proceed with the installation.

Open PyCharm. Go to setting under the configure. In the setting, go to the project interpreter. On the top right, press the gear icon and press add. **Go to Conda Environment**, **Existing Environment**, and you should see the created environment similar to this
![PyCharm-add-interpreter](https://user-images.githubusercontent.com/85147048/120616430-6c336700-c483-11eb-92e0-cbe414facb59.png)

Now you can create a new project under the built environment.
![PyCharm Create a new project](https://user-images.githubusercontent.com/85147048/120617118-05fb1400-c484-11eb-9930-8f0820e22a29.png)

To Check if you have successfully installed PyTorch, you can try these following code.

```python
import torch
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    
x = torch.rand(5, 3)
print(x)
```

# Basic PyTorch
## Tensors
Tensors are a specialized data structure similar to arrays and matrixes. Tensors are similar to [NumPy's](https://numpy.org/devdocs/user/absolute_beginners.html) ndarrays. However, Tensors can run on GPUs or hardware accelerators, making it significantly faster than NumPy, especially when running on GPUs. You can learn more about [NumPy vs. Tensors](https://medium.com/thenoobengineer/numpy-arrays-vs-tensors-c58ea54f0e59) here. A concise definition of a tensor is a matrix of N ≥ 3 dimensions. Note that all tensors are immutable, like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

### Initializing a Tensor
You can initialize a tensor in many ways. Here is the reference source, [TENSORS](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

#### Directly with oparetor
Tensor can be initialized using ```torch.tensor()``` creating a specific tensor.
```python
t1 = torch.tensor([[1, 2],
                  [3, 4]])
t2 = torch.tensor([[5, 6],
                  [7, 8]])
print("t1: \n", t1)
print("t2: \n", t2)
```
*output*
```
t1: 
 tensor([[1, 2],
        [3, 4]])
t2: 
 tensor([[5, 6],
        [7, 8]])
```
#### Directly from arrays
Tensors can be initialized from a created array. The data type is automatically inferred.
```python
data = [[1, 2], [3, 4]]
tensor_data = torch.tensor(data)
print(f"tensor_data from arrays: \n {tensor_data} \n")
```
#### From a NumPy Array
Tensors can be created from NumPy arrays and vice versa.
```python
# Tensor from Numpy
data = [[1, 2], [3, 4]]
np_array = np.array(data)
tensor_from_np = torch.from_numpy(np_array)
print(f"From Numpy: \n {tensor_from_np} \n")

# NumPy from Tensor
np_from_tensor = np.array(tensor_from_np)
print(f"From Tensor: \n {np_from_tensor} \n")
```
#### From another tensor
The newly created tensor retains the properties: shape and datatype of the argument tensor unless explicitly overridden.
```python
tensor_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {tensor_ones} \n")

tensor_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {tensor_rand} \n")
```
#### With a random/constant values
*shape* is a tuple of tensor dimensions. You can initialize a tensor with any constant value or random numbers. <br/>
*rand* is random.
```python
shape = (4, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")
```
### Tensor's Attributes
Attributes describe their shape, datatype, and the device on which the tensor is stored.<br/>
```tensor.shape``` will show the dimension of the tensor.<br/>
```tensor.dtype``` will show the datatype of the tensor.<br/>
```tensor.device``` will show the device the tensor is stored on.<br/>

```python
tensor = torch.rand(3, 4,)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
*output*
```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### Tensor's Dimension/shape
Since dimensions have been mentioned multiple times, here is some information regarding them. This section will help you visualizing multidimensions tensor/arrays. You can also read more about it here [Understanding Dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be).

#### Rank-0 or Scalar
A *scalar* contains a single value and has no axes.
```python
import torch
rank_0_tensor = torch.tensor(4)
print(rank_0_tensor)
print(rank_0_tensor.shape)
```
*output*
```
tensor(4)
torch.Size([])
```
#### Rank-1 or Vector
A *vector* tensor is a list of values and has only one axis.
```python
import torch
rank_1_tensor = torch.tensor([1, 2, 3, 4, 5, ])
print(rank_1_tensor)
print(rank_1_tensor.shape)
```
*output*
```
tensor([1, 2, 3, 4, 5])
torch.Size([5])
```
#### Rank-2 or Matrix
A *matrix* or *rank-2* tensor has two axes like a 2 dimesional arrays.
```python
import torch
rank_2_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print(rank_2_tensor)
print(rank_2_tensor.shape)
```
*output*
```
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
torch.Size([4, 2])
```
#### Rank-3 or 3 Dimesionals
Tensor with 3 axes.
```python
import torch
rank_3_tensor = torch.tensor(
    [[[0, 1, 2, 3, ],
      [4, 5, 6, 7, ]],
     [[8, 9, 10, 11, ],
      [12, 13, 14, 15, ]],
     [[16, 17, 18, 19, ],
      [20, 21, 22, 23]], ])
print(rank_3_tensor)
print(rank_3_tensor.shape)
```
*output*
```
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7]],

        [[ 8,  9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23]]])
torch.Size([3, 2, 4])
```
![3Dimensions-1](https://user-images.githubusercontent.com/85147048/120790992-d15b8b00-c55d-11eb-9487-6ce3cb3ca0b3.jpg)

It is easier to construct the multidimensional tension with the last element of the shape/size. In this example (tensor size [3, 2, 4]), you start with 4 elements on an axis, 2 on another axis becoming 4 by 2 tension, and 3 on the last axis becoming 4 by 2 by 3 tension. In addition, it's easier to keep track of your multidimensional tensions when you keep the same format consistently.
For example, construct starting on the x-axis, y-axis, z-axis, then x-axis again.

#### Rank-4 tensor, and higher.
Basically a stack of the matrix tensors.
```python
import torch
rank_4_tensor = torch.zeros([3, 2, 2, 3, ])
print(rank_4_tensor)
print(rank_4_tensor.shape)
```
*output*
```
tensor([[[[0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.]]],


        [[[0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.]]],


        [[[0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.]]]])
torch.Size([3, 2, 2, 3])
```
![4+Dimensions](https://user-images.githubusercontent.com/85147048/120795255-73ca3d00-c563-11eb-8ba9-19736313a134.jpg)

### Tensor's Operations
There are over a hundred tensor oparetions, including arithmetic, linear algebra, matrix manipulation, sampling, and more [here](https://pytorch.org/docs/stable/torch.html).


#### Tensor on CUDA/CPU
Since we have talked about CUDA in the installation section, we can move our tensor to GPU if available. By default, tensors are created on the CPU. However, you can move run them on GPU at a higher speed than on a CPU.

```python
tensor_cpu = torch.rand([2, 2, 2, ])

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor_cuda = tensor_cpu.to('cuda')

print(tensor_cuda)
```
*output*
```
tensor([[[0.0510, 0.7120],
         [0.4976, 0.6011]],

        [[0.3774, 0.0082],
         [0.5302, 0.2511]]], device='cuda:0')
```
*note* device='cuda:0' is your GPU index at 0. Useful when you have multiple GPUs.

#### Standard numpy-like indexing and slicing
Access, print, or edit different indexes.
A coding example from [PyTorch](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)<br/>
```python
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```
*output*
```
First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```
More example codes in the included files/codes.

#### Joining Tensors
*torch.stack* **stacks** a sequence of tensors along a **new dimension**<br/>
*torch.cat* con**cat**enates the sequence of tensors in the **given dimension.**<br/>
```python
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
```
*output*
```
stack: 
 tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
stack dimension:  torch.Size([2, 2, 2])

cat | dim=0: 
 tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
cat | new dimension:  torch.Size([4, 2])

cat | dim=1: 
 tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
cat | new dimension:  torch.Size([2, 4])
```
So if **A** and **B** are of shape (4, 5), torch.cat((A, B), dim=0) will be of shape (8, 5), and torch.stack((A, B), dim=0) will be of shape (2, 4, 5).

#### Arithmetic operations
#### Dot Multiplication
```torch.mul(a, b)``` is a multiplication of the corresponding bits of matrix a and b. The dimensions of the two metrix are generally equal (ex: the number of elements have to match) The output metrix will keep its shape/dimension.
```python
# dot multiplication
t1 = torch.randn(1, 2, )
t2 = torch.randn(1, 2, )

tMul = torch.mul(t1, t2)

print("t1: \n", t1)
print("t2: \n", t2, "\n")

print("dot multiplication: \n", tMul, "\n")
```
*output*
```
t1: 
 tensor([[0.8898, 1.2521]])
t2: 
 tensor([[ 1.0311, -0.5143]]) 

dot multiplication: 
 tensor([[ 0.9175, -0.6440]])
```
#### Matrix Multiplication
```torch.mm(a, b)``` multiplies the matrix a and b.
```python
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
```
*output*
```
t1: 
 tensor([[1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]]) 
 torch.Size([3, 4])
t2: 
 tensor([[1, 2],
        [1, 2],
        [1, 2],
        [1, 2]]) 
 torch.Size([4, 2])
matrix multiplication: 
 tensor([[10, 20],
        [10, 20],
        [10, 20]]) 
 torch.Size([3, 2])
```

```torch.matmul(a, b)``` A high-dimensional matrix multiplication.
```python
# torch.matmul(a, b)
t1 = torch.ones(2, 4, 2)
t2 = torch.ones(2, 2, 3)
tMatmul = torch.matmul(t1, t2)
print("matrix multiplication: \n", tMatmul, "\n"
      , tMatmul.shape)
```
*output*
```
matrix multiplication: 
 tensor([[[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]],

        [[2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.],
         [2., 2., 2.]]]) 
 torch.Size([2, 4, 3])
```
There are many more operations such as: <br/>
```tensor.sum()``` to sum all the elements into a single tensor value.<br/>
```tensor.item()``` to change the tensor value into Python numerical value like float.<br/>
```tensor.add_(x)``` to add all the elements with **x**.<br/>
note: **_** suffix is called **In-Place operations**. Operations that store the result into the operand are called in-place. Basically you are chaning/altering the variable. For example x.copy_(y) or x.t_() will change the x.

## Datasets & Dataloaders
Processing data samples. PyTorch provides operators that help readability and modularity. You can use pre-loaded datasets provided by PyTorch or your own datasets. **Dataset** stores the samples and their corresponding labels while **DataLoader** wraps an iterable around the **dataset** to enable easy access to the samples. **DataLoader** comes into handy when the datasets become prominent and are required to be loaded into memory at once.  **DataLoader** parallelizes the data loading process with the support of automatic batching.

```torch.utils.data.DataLoader```
```torch.utils.data.Dataset```

An example from [PyTorch's Datasets/Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), we are going to load a dataset from the **Fashion-MNIST** which is one of subclasses of ```torch.utils.data.Dataset``` [TORCHVISION.DATASETS](https://pytorch.org/vision/stable/datasets.html#fashion-mnist).


