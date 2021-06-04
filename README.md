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
Since dimensions have been mentioned multiple times, here is some information regarding them.
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

