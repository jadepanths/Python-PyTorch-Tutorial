# Python-PyTorch-Tutorial
Basic PyTorch Tutorials in Python for beginners who have not used Python for years.

# Work In Progress

# Installation
Install an IDE on your local device like [IDLE](https://www.python.org/downloads/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows), or [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/features/python/).  I personally recommend [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) as it has an excellent debugger, sophisticated autocompletion, and refactoring support.

Install [Anaconda](https://www.anaconda.com/products/individual). Anaconda is an toolkit that include open-source packages and libraries. You will need this to install libraries and manage environments where you will compile your code in.

Check if your computer/labtop is ***CUDA-capable*** CUDA is limited to ***NVIDIA*** GPU. If you know what GPU you have, you can simply check if it CUDA capable here [CUDA GPUS LIST](https://developer.nvidia.com/cuda-gpus). If you do not know what GPU do you have, Open "Run" in the window search bar or the start menu. Then run this command: ```control /name Microsoft.DeviceManager```. Then go to your Display Adapters to check your GPUs.
![Device Manager](https://user-images.githubusercontent.com/85147048/120597340-1eacff00-c46f-11eb-824c-7fcfffb5f5ee.png)
After making sure you have a CUDA-capable gpu, you can download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Install PyTorch
First, create a new environment by openning Anaconda Prompt in the window start menu. Then, run this command: ```conda create --name deeplearning``` where "deeplearning" is the name of the enviroment and can use any name you wish. The terminal will ask you to preceed, tpye 'y' and enter. Then, run the following command ```conda activate deeplearning``` (note: "deeplearning" is the environment name you created. Makde sure to use the same name).<br/>
For **NON-CUDA** run this command <br/>```conda install pytorch torchvision torchaudio cpuonly -c pytorch```.<br/>
For **CUDA CUDA-capable** run this command <br/>```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```.<br/>
Then run 'y' to proceed the installation.

Open PyCharm. Go to setting under the configure. In the setting, go to project interpreter. On the top right, press the gear icon and press add. **Go to Conda Environment**, **Existing Enviromnet**, and you should see the created enviromnet similar to this
![PyCharm-add-interpreter](https://user-images.githubusercontent.com/85147048/120616430-6c336700-c483-11eb-92e0-cbe414facb59.png)

Now you can create a new project under the created enviroment.
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
Tensors are a specialized data structure similar to arrays and metrices. Tensors are similar to [NumPy's](https://numpy.org/devdocs/user/absolute_beginners.html) ndarrays. However, Tensors can run on GPUs or hardware accelerators making significant faster than NumPy especially when running on GPUs. You can learn more about [NumPy vs Tensors](https://medium.com/thenoobengineer/numpy-arrays-vs-tensors-c58ea54f0e59) here. A very short definition of tensor is a metrix of N ≥ 3 dimensions. Note that all tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.

### Initializing a Tensor
#### Directly from arrays
Tensors can be initialized from a created array. The data type is automatically inffered.
```python
data = [[1, 2], [3, 4]]
tensor_data = torch.tensor(data)
print(f"tensor_data from arrays: \n {tensor_data} \n")
```
#### From a NumPy Array
Tensors can be created from NumPy arrays and vice versa
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
