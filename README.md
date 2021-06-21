# Python-PyTorch-Tutorial
Basic PyTorch Tutorials in Python for beginners

# Work in Progress

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

# [Installation](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Installation.ipynb)
This will walk you through the PyTorch installation, including installing python and a toolkit.

# [Tensors](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb)
A neural network is a mathematical function. It takes in one or multiple inputs, process it and produces one or more outputs, In PyTorch, neural networks are composed of PyTorch tensors. <br/>

Tensors are a specialized data structure similar to arrays and matrixes. Tensors are similar to [NumPy's](https://numpy.org/devdocs/user/absolute_beginners.html) ndarrays. However, Tensors can run on GPUs or hardware accelerators, making it significantly faster than NumPy, especially when running on GPUs. You can learn more about [NumPy vs. Tensors](https://medium.com/thenoobengineer/numpy-arrays-vs-tensors-c58ea54f0e59) here. A concise definition of a tensor is a matrix of N ≥ 3 dimensions. Note that all tensors are immutable, like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

## [Initializing a Tensor](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=SxOfpMUgMaqO)
You can initialize a tensor in many ways. Here is the reference source, [TENSORS](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).

- [Directly With Operator](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=SxOfpMUgMaqO)
- [Directly From Arrays](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=cNs-l1FpPfig)
- [From a NumPy Array](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=KUrpxs7UP6Vh)
- [From Another Tensor](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=MyKNEnufRZgH)
- [With a Random/Constant Values](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=n9NiDSDAAR4W)

## [Tensor's Attributes](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=XRK7OefMBwkF)
Attributes describe their shape, datatype, and the device on which the tensor is stored.

## [Tensor's Dimension/shape](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=IGoAIHHYCZQi)
-   [Scalar](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=HuOk1wPnC2KR&line=1&uniqifier=1)
-   [Vector](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=YgdBXldaDuRz)
-   [Metrix](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=A_lQYUuPEPzU)
-   [3 Dimensionals](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=MYq4_iCnEXoS)
-   [4+ Dimensionals](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=Hd2k20WtFOLC)

## [Tensor's Operations](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=z7v6RwbFGEsD)
-   [Tensor on CUDA/GPU](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=7eqGXHENGJ4P)
-   [Indexing and Slicing](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=tvC7M5tQH3z0&line=1&uniqifier=1)
-   [Joining Tensors](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=1kzgb2X1JA62)

### [Arithmetic Operations](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=pHMBEfUuNr53)
-   [Dot Multiplication](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=8_RPGxrTNs7J)
-   [Matrix Multiplication](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=Fyspiq90OGcM)

## [Tensor Memory Location](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Tensors.ipynb#scrollTo=zhwLfsFskqmN)
This is where you have to be careful when comverting and modifying tensors. As they often point to the same memory address. Like a C++ pointer, when you modify one variable, another variable will be modified as well.

# [Datasets & Dataloaders](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=qEKwxdHehxBW)
-   [Loading a Dataset](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=_3d3gXKMiNpw)
-   [Iterating and Visualizing the Dataset](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=_m98jXZ-tRsC)
-   [Creating Custom Dataset](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=J6BPQZ5JtqMu)
-   [Preparing Data for training with DataLoaders](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=LHxDDLYCumog)
-   [Iterate through the DataLoader
](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Datasets_and_Dataloaders.ipynb#scrollTo=N5IGo2s3u2Bg)


# [Transforms](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Transforms.ipynb#scrollTo=v0-5oX6Qx62n)

-   [ToTensor()](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Transforms.ipynb#scrollTo=yNZsjuwEyHNy)
-   [Lambda Transforms](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Transforms.ipynb#scrollTo=MVhTKLLAyJEL)
-   [One-hot Encoded Tensor](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Transforms.ipynb#scrollTo=zWCqL5QiySr2)


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
