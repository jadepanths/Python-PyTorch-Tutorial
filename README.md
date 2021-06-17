# Python-PyTorch-Tutorial
Basic PyTorch Tutorials in Python for beginners

# Disclaimer
This repository contains many resources I found during my study of this subject. This repository is a summary, note, resource, and information I found useful and would like to share with fellow beginners. I have no means of claiming the credits to myself, and I will be trying to cite as many things as possible.

## Background and Recommendation
### Mathematics
To master deeplearning and AI, according to [Andrew Ng](https://en.wikipedia.org/wiki/Andrew_Ng), these areas of math are the most imporant, in decreasing order:
1. Linear Algebra
2. Probability and Statistics
3. Calculus (including multivariate calculus)
4. Optimization

My recommendation is to learn basic linear algebra then basic programming.
### Basic Python
There are many online classes and toturials you can find online. 

# Work In Progress

# Installation
## Install an IDE (Windows, macOS, and Linux)
Install an IDE on your local device like [IDLE](https://www.python.org/downloads/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), or [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/features/python/). I personally recommend [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) as it has an excellent debugger, sophisticated autocompletion, and refactoring support. Moreover, PyCharm is a cross-platform IDE that works great on the **Windows**, **macOS**, and **Linux**.

## Install a toolkit (Windows, macOS, and Linux
Install [Anaconda](https://www.anaconda.com/products/individual). Anaconda is a toolkit that includes open-source packages and libraries. You will need this to install libraries and manage environments where you will compile your code in. 

## Check CUDA Capability
Check if your computer/labtop is ***CUDA-capable*** CUDA is limited to ***NVIDIA*** GPU. If you know what GPU you have, you can simply check if it CUDA capable here [CUDA GPUS LIST](https://developer.nvidia.com/cuda-gpus). <br/> <br/> 
If you do not know what GPU do you have, check the following steps. <br/> <br/> 
For **Windows**, open "Run" in the window search bar or the start menu. Then run this command: ```control /name Microsoft.DeviceManager```. Then go to your Display Adapters to check your GPUs. <br/>
![Device Manager](https://user-images.githubusercontent.com/85147048/120597340-1eacff00-c46f-11eb-824c-7fcfffb5f5ee.png) <br/>

For **MacOS**, under the Apple menu, select _About This Mac_, click the _More Indo..._ botton, and select _Graphics/Displays_ under the Hardware list. You will find the vendor name and model of your graphics card. You can also check version of the CUDA, if have one installed, by running ```nvcc -V``` in a terminal window. <br/>

For **Linux**, you can use GUI to identify the graphics card. You can open the _setting_ dialog, and then click _Details_ in the sidebar. In the _About_ panel, look for a _Graphics_ entry. Here is an example.
![16-5](https://user-images.githubusercontent.com/85147048/122430600-f3252b00-cfbd-11eb-8ba5-1fc9c9d99cbf.png) <br/>

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
# Tensors
A neural network is a mathematical function. It takes in one or multiple inputs, process it and produces one or more outputs, In PyTorch, neural networks are composed of PyTorch tensors. <br/>


Tensors are a specialized data structure similar to arrays and matrixes. Tensors are similar to [NumPy's](https://numpy.org/devdocs/user/absolute_beginners.html) ndarrays. However, Tensors can run on GPUs or hardware accelerators, making it significantly faster than NumPy, especially when running on GPUs. You can learn more about [NumPy vs. Tensors](https://medium.com/thenoobengineer/numpy-arrays-vs-tensors-c58ea54f0e59) here. A concise definition of a tensor is a matrix of N ≥ 3 dimensions. Note that all tensors are immutable, like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

## Initializing a Tensor
You can initialize a tensor in many ways. Here is the reference source, [TENSORS](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

### Directly with oparetor
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
### Directly from arrays
Tensors can be initialized from a created array. The data type is automatically inferred.
```python
data = [[1, 2], [3, 4]]
tensor_data = torch.tensor(data)
print(f"tensor_data from arrays: \n {tensor_data} \n")
```
### From a NumPy Array
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
### From another tensor
The newly created tensor retains the properties: shape and datatype of the argument tensor unless explicitly overridden.
```python
tensor_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {tensor_ones} \n")

tensor_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {tensor_rand} \n")
```
### With a random/constant values
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
## Tensor's Attributes
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

## Tensor's Dimension/shape
Since dimensions have been mentioned multiple times, here is some information regarding them. This section will help you visualizing multidimensions tensor/arrays. You can also read more about it here [Understanding Dimensions in PyTorch](https://towardsdatascience.com/understanding-dimensions-in-pytorch-6edf9972d3be).

### Rank-0 or Scalar
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
### Rank-1 or Vector
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
### Rank-2 or Matrix
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
### Rank-3 or 3 Dimesionals
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

### Rank-4 tensor, and higher.
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

## Tensor's Operations
There are over a hundred tensor oparetions, including arithmetic, linear algebra, matrix manipulation, sampling, and more [here](https://pytorch.org/docs/stable/torch.html).


### Tensor on CUDA/CPU
Since we have talked about CUDA in the installation section, we can move our tensor to GPU if available. By default, tensors are created on the CPU. However, you can move run them on GPU at a higher speed than on a CPU.

#### Example 1
```python
import torch
tensor_cpu = torch.rand([2, 2, 2, ])

# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor_cuda = tensor_cpu.to('cuda')
    # .to move the tensor to your gpu

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

#### Example 2
```python
import torch

if torch.cuda.is_available():
    # Set the cuda0 to be the first GPU (index 0)
    cuda0 = torch.device("cuda")
    
    # cuda1 = torch.device("cuda:1) # second and more GPUs if available
    # Cross-GPU operations are not allowed by default.
    
    x = torch.ones(3, device=cuda0)
    y = torch.ones(3)
    y = y.to(cuda0) # Move tensor y to GPU
    
    # This will be performed on the GPU 
    z = x + y
    
    # z.numpy() will not work as it can handle only CPU tensor 
    # Would have to move it back to CPU if you would like to convert
    z = z.to("cpu")
 
    print(x)
    print(y)
    print(z)
```
*output*
```
tensor([1., 1., 1.], device='cuda:0')
tensor([1., 1., 1.], device='cuda:0')
tensor([2., 2., 2.])
```

### Standard numpy-like indexing and slicing
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

### Joining Tensors
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

### Arithmetic operations
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

## Tensor Memory Location
This is where you have to be careful when comverting and modifying tensors.
As they often point to the same memory address. Like a C++ pointer, when you modify one variable, another variable will be modified as well.

```Python
import torch

a = torch.ones(3)
print(a)
b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)
```
_output_
```
tensor([1., 1., 1.])
[1. 1. 1.]
tensor([2., 2., 2.])
[2. 2. 2.]
```
You can see that when modify **a** with .add_(1), **b** will be modified as well.
The reason is that **a** and **b** both point to the same memory address. The same goes to this following example.

```Python
import torch
import numpy as np

a = np.ones(3)
print(a)
b = torch.from_numpy(a)
print(b)

a += 1
print(a)
print(b)
```
_output_
```
[1. 1. 1.]
tensor([1., 1., 1.], dtype=torch.float64)
[2. 2. 2.]
tensor([2., 2., 2.], dtype=torch.float64)
```

# Datasets & Dataloaders
Processing data samples. PyTorch provides operators that help readability and modularity. You can use pre-loaded datasets provided by PyTorch or your own datasets. **Dataset** stores the samples and their corresponding labels while **DataLoader** wraps an iterable around the **dataset** to enable easy access to the samples. **DataLoader** comes into handy when the datasets become prominent and are required to be loaded into memory at once.  **DataLoader** parallelizes the data loading process with the support of automatic batching.

**DataLoader:** ```torch.utils.data.DataLoader```<br/>
**Dataset:** ```torch.utils.data.Dataset```

## Loading a Dataset

Here are examples from [PyTorch's Datasets/Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), we are going to load a dataset from the **Fashion-MNIST** which is one of subclasses of ```torch.utils.data.Dataset```, [TORCHVISION.DATASETS](https://pytorch.org/vision/stable/datasets.html#fashion-mnist).

**Parameters**<br/>
```root``` is the path where the train/test data is stored. <br/>
```train``` specifies training or test dataset. <br/>
```downlad=true``` downloads the data from the internet if it's not avaliable at ```root```. <br/>
```transform``` and ```target_transform``` specify the featre and label transformations. <br/>

```Python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```
_output_
```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Processing...
Done!
```
## Iterating and Visualizing the Dataset
We can index Datasets manually like a list using ```training_data[index]```. We use **matplotlib** to render some samples in our training data.
```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```
_output_ <br/>
![Figure_1](https://user-images.githubusercontent.com/85147048/121321204-17d33000-c938-11eb-840b-8b0ad1634074.png)

## Creating Custom Dataset
A custom Dataset must implement these three functions: ```__int__```, ```__len__```, abd ```__getitem__```.

An implementation from the FashionMNIST image are stored in a dirctory *img_dir*, and thier labels are stored sperately in a CSV file *annotations_file*. You can read the code [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files).

### __int__
This function is a reseved method. It is called as a constructor in object oriented terminology. We initialize the directory containing the images, the annotations file, and both transforms.

### __len__
return the number of sameples in our dataset.

### __getitem__
This function loads and returns a sample from the dataset at a given index.
Based on the index, it identifies the image's location on disk, converts that to a tensor using ```read_image```, retrieves the corresponding label from the cvs data, calls the transform functioins on them(if applicable), and returns the tensor image and corresponding label in a tuple.

## Preparing Data for training with DataLoaders
The *Dataset* retrieves our dataset's features and labels one sample at a time, We can use python's multiprocessing to speed up data retrieval. We also want to pass samples in minibatches, reshuffle the data at every epoch to reduce model overfitting.
_note_ Overfitting is an error that occurs in data modeling as a result of a particular function aligning too closely to a minimal set of data points (overely complex model).

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## Iterate through the DataLoader
Once we have loaded the dataset into the _Dataloader_, we can iterate through the dataset. Each iteration below returns a batch of *train_feature* and *train_labels*. When spicifying _shuffle=true_, we shuffle all the data after the iteration is over.

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
# Transforms
Data does not always come in its final processed form and is required for traning machine learning algorithms. We use **transforms** to preform some manupulation of the data and make it suitable for traning like a raw ingredient where we need to cook it.
<br/>
<br/>
All TorchVision datasets have two parameters - transform_ to modify the features and _target_transform_ to modify the labels - that accept callables containing the transformation logic. There are many commnon transforms here, [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html).
<br/>
<br/>
Using the example from pytorch.org (FashionMNIST), the FashionMNIST features are in PIL image format, and the labels are intergers. For training, we need the feature as normalized tensors, and the labels as one-hot encoded tensors. 

```python
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
## ToTensor() 
Converts a PIL image or NumPy _ndarray_ into a _FloatTensor_ and scale the image's pixel intensity values in the range [0.0 to 1.0]. The works for the image with elements that are in range from 0 to 255.

## Lambda Transforms
apply any user-defined lambda function. Here, we difine a function to turn integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and called [scatter_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_) which assigns a value=1 on the index as given by the label y.

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

## One-hot Encoded Tensor
Since we have mentioned one-hot encoded tensor several times, we are going to take a look what it actually is. [Here](https://datascience.stackexchange.com/questions/30215/what-is-one-hot-encoding-in-tensorflow) is a explaination from Djib2011 on a Stack.Exchange question.
<br/>
Suppose we have a catagorical feature in our dataset like colour. Your samples can only be either red, yellow, or blue. In machine learning algorith, we have to pass the argument as a number instead of strings. So we mapped the colours like this: <br/>
<br/>
red --> 1 <br/>
yellow --> 2 <br/>
blue --> 3 <br/>
<br/>
We have replaced the string with the mapped value. However, this method can create negative side effects in our model when dealing with numbers. For example, blue is larger than yellow because 3 is larger than 2. Or red and yellow combied is equal to blue because of 1 + 2 = 3. The model has no way of knowing that these data was catagorical and then were mapped as intergers.<br/> <br/>
Now is where **one-hot encoding** comes in handy. We create *N* **new features**. where *N* is the number of unique values in the orignal feature, where _N_ is the number of unique values in the original feature. In our example, _N_ would be eqaul to 3 as we only have 3 unique colours: red, yellow, and blue. <br/>
<br/>
Each of these features is binary and would correspond to **one** of these unique values. In our example, the first feature would be a binary feature tellinus if that samle is red or not. The second would be the same this for yellow, and the Third for blue. <br/>
<br/>
An example of such a transformation is illustrated below: <br/>
![mtimFxh](https://user-images.githubusercontent.com/85147048/121554816-a9c46100-ca3c-11eb-9b19-9bfefe159680.png)<br/>
Note, that because this approach increases the dimensionality of the dataset, if we have a feature that takes many unique values, we may want to use a more sparse encoding.

# Build The Neural Network
The Neural networks comprise of layer or modules that perform operations on data. The [torch.nn](https://pytorch.org/docs/stable/nn.html) namespace provides all the building blocks you need to build your own neural network. All the modules in PyTorch subclasses the [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). A neural network is modile itself that consists of other modules/laters. This nested structure allows for building and managing complex architectures easily.<br/>
<br/>
We will build a neural network to classify images in the FasionMNIST dataset.
```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## Get Device for training
Training a model on a hardware accelerator like a GPU is faster than traing on a cpu. Therefore, we should check if cuda is available. 
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
```
_output_
``` Using cuda device```

## Define the Class
We define our neural network bu subclassing _nn.Module_, and initialize the neural network laters in ```__init__```. Every _nn.Module_ subclass implements the operations on input data in the _forward_ method.

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```
After creating an instance if _NeuralNetwork_, we can move it to the _device_ and print its structure.
```python
model = NeuralNetwork().to(device)
print(model)
```
_output_
```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
)
```
Note: ReLu is like Riemann sums. You can approximate any continuos functions with lots of little reactangles. ReLu activations can produced lots of little rectangles. ReLu can make complicated shapes and approximate maby complicated domains.

To use the model, we pass it the input data. This executes the model’s _forward_, along with some [background operations](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866). Do not call model.forward() directly. <br/>
<br/>
Calling the model on the input returns a 10-dimensional tensor with a raw predicted values for each class. We get the prediction probabilities by passign ti through an instance of the _nn.Softmax_ module.

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```
_output_
```Predicted class: tensor([8], device='cuda:0')```

## Model Layers
Break down the layers in the FashionMNIST model. We will take a sample minibatch of 3 images of size 28x28.

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```
_output_
```torch.Size([3, 28, 28])```

### Flatten
We initialize the [nn.Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) layer to convert each 2D 28*28 image into a contiguous of 784 pixel values (the minibatch dimension (at dim=0) is maintained)

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```
_output_
```torch.Size([3, 784])```

### Linear
The [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) is a module that applies a linear transformation on the input using its stored weights and biases.
```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```
_output_
```torch.Size([3, 20])```

### ReLU
[nn.ReLu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) is used between the linear layers. (There is other activations to introduce non-linearity in your model.

Non-linear activations create the complex mapping between the model's input and outputs. They are apllied after the linear transformation to introduce _nonlinearity_.
```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```
_output_
```
Before ReLU: tensor([[-0.2028, -0.3458,  0.4659, -0.6775,  0.1644, -0.3828, -0.0769, -0.0652,
          0.3995,  0.0828, -0.1919, -0.3055, -0.2934, -0.0490,  0.1276, -0.1160,
          0.0321, -0.1210, -0.4174, -0.2444],
        [ 0.0683, -0.4251,  0.6049, -0.7045,  0.4218, -0.2934, -0.3552,  0.1145,
          0.2328,  0.1044, -0.1296, -0.4870, -0.4180, -0.2836,  0.1672, -0.2017,
         -0.1182, -0.3317, -0.0066, -0.2201],
        [-0.2765, -0.1589,  0.6315, -0.6072,  0.1345, -0.3009, -0.0169,  0.0051,
          0.7639, -0.0118,  0.1058,  0.0022, -0.5412, -0.0155,  0.0434, -0.3487,
         -0.2751, -0.4741, -0.6828, -0.0236]], grad_fn=<AddmmBackward>)


After ReLU: tensor([[0.0000, 0.0000, 0.4659, 0.0000, 0.1644, 0.0000, 0.0000, 0.0000, 0.3995,
         0.0828, 0.0000, 0.0000, 0.0000, 0.0000, 0.1276, 0.0000, 0.0321, 0.0000,
         0.0000, 0.0000],
        [0.0683, 0.0000, 0.6049, 0.0000, 0.4218, 0.0000, 0.0000, 0.1145, 0.2328,
         0.1044, 0.0000, 0.0000, 0.0000, 0.0000, 0.1672, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 0.6315, 0.0000, 0.1345, 0.0000, 0.0000, 0.0051, 0.7639,
         0.0000, 0.1058, 0.0022, 0.0000, 0.0000, 0.0434, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000]], grad_fn=<ReluBackward0>)

```
### Sequential
[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) is an ordered container of modules. This data is passed through all the modules in the same order as defined. You can use sequential containers to put together a quicl network like *seq_modules*
```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```
