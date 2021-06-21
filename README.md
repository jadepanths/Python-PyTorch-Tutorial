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
A neural network is a mathematical function. It takes in one or multiple inputs, process it and produces one or more outputs, In PyTorch, neural networks are composed of PyTorch tensors.

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


# [Build The Neural Network](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=tFO5mydrz7e1)
-   [Get Device for training](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=8jSMtV4H0DA6)
-   [Define the Class](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=loUv67Xx0tY2)
## [Model Layers](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=gAduRjFV2HsG)
-   [Flatten](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=CT_8OIop2MFD)
-   [Linear](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=cH-nXbGQ2SaH)
-   [ReLU](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=gszx6zb-2xvS)
-   [Sequential](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=eWSEnPsc27Zz)
-   [Softmax](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=wPWxc6gi09LG)
-   [Model Parameters](https://colab.research.google.com/github/jadepanths/Python-PyTorch-Tutorial/blob/main/Build_The_Neural_Network.ipynb#scrollTo=SwZl_oAW2eXp)

