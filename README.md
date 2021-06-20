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

# [Installation](https://github.com/jadepanths/Python-PyTorch-Tutorial/blob/main/Installation.ipynb)
This will walk you through the PyTorch installation, including installing python and a toolkit.

# Tensors
A neural network is a mathematical function. It takes in one or multiple inputs, process it and produces one or more outputs, In PyTorch, neural networks are composed of PyTorch tensors. <br/>

Tensors are a specialized data structure similar to arrays and matrixes. Tensors are similar to [NumPy's](https://numpy.org/devdocs/user/absolute_beginners.html) ndarrays. However, Tensors can run on GPUs or hardware accelerators, making it significantly faster than NumPy, especially when running on GPUs. You can learn more about [NumPy vs. Tensors](https://medium.com/thenoobengineer/numpy-arrays-vs-tensors-c58ea54f0e59) here. A concise definition of a tensor is a matrix of N ≥ 3 dimensions. Note that all tensors are immutable, like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

## [Initializing a Tensor](https://colab.research.google.com/drive/1Fv6-hEuDmCJvUltMVJuSJXcffqJmuFcU#scrollTo=GMW11qS2FFCW&line=2&uniqifier=1)
You can initialize a tensor in many ways. Here is the reference source, [TENSORS](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html).
- [Directly With Operator](https://colab.research.google.com/drive/1Fv6-hEuDmCJvUltMVJuSJXcffqJmuFcU#scrollTo=SxOfpMUgMaqO)
- [Directly From Arrays](https://colab.research.google.com/drive/1Fv6-hEuDmCJvUltMVJuSJXcffqJmuFcU#scrollTo=cNs-l1FpPfig)
- [From a NumPy Array](https://colab.research.google.com/drive/1Fv6-hEuDmCJvUltMVJuSJXcffqJmuFcU#scrollTo=KUrpxs7UP6Vh)


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
