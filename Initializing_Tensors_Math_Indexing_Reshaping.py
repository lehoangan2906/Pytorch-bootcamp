import torch
import numpy as np

# initializing tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float32, device = device, requires_grad = True)

print(my_tensor)    # shows the tensor

print(my_tensor.dtype)  # shows the data type

print(my_tensor.device) # shows the device on which the tensor is stored

print(my_tensor.shape)  # shows the shape of the tensor

print(my_tensor.requires_grad)  # show if the tensor is a gradient or not


# other common initialization methods

# creates a tensor with size (3, 3) and all elements are uninitialized and have values as the values of the memory blocks
x = torch.empty(size = (3, 3)) 

# creates a tensor with size (3, 3) and all elements are initialized with 0
x = torch.zeros((3, 3))  

# creates a tensor with size (3, 3) and all elements are initialized with random values between 0 and 1
x = torch.rand((3, 3))  

# creates a tensor with size (3, 3) and all elements are initialized with 1
x = torch.ones((3, 3))  

# creates a tensor with size (5, 5) and all elements are initialized with 1 except the diagonal elements are initialized with 0
x = torch.eye(5, 5)

# creates a tensor with size (5,) and all elements are initialized with 0 and 1, 2, 3, 4, 5
x = torch.arange(start = 0, end = 5, step = 1) 

# creates a tensor with size (10,) and all elements are initialized with 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
x = torch.linspace(start = 0.1, end = 1, steps = 10) 

# creates a tensor with size (1, 5) and all elements are initialized with random values with mean 0 and standard deviation 1
x = torch.empty(size = (1, 5)).normal_(mean = 0, std = 1) 

# creates a tensor with size (1, 5) and all elements are initialized with random values with mean 0 and standard deviation 1
x = torch.empty(size = (1, 5)).uniform_(0, 1) 

# creates a tensor with size (3, 3) and all elements are initialized with 1 except the diagonal elements are initialized with 0
x = torch.diag(torch.ones(3))  


# how to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)

print(tensor)

print(tensor.bool())    # converts the tensor to boolean

print(tensor.short())   # converts the tensor to short

print(tensor.long())    # converts the tensor to long

print(tensor.half())    # converts the tensor to half

print(tensor.float())   # converts the tensor to float

print(tensor.double())  # converts the tensor to double -- float64


# Array to Tensor conversion and vice-versa

np_array = np.zeros(5)
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()



# =========================================================================== #
#                      Tensor Math & COmparation Operations                   #     
# =========================================================================== # 

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out = z1)
print(z1)

z = x + y
print(z)

# subtraction

z = x - y
print(z)

# Division
z = torch.true_divide(x, y) # elementwise division if they are equal shapes

# Inplace operations
t = torch.zeros(3)
t.add_(x)   # the operation is done inplace, which means the original tensor is modified
t += x  

# Exponentiation
z = x.pow(2)    # elementwise power
print(z)

z = x**2    # elementwise power


# Simple comparison operations
z = x > 0
print(z)

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))

x3 = torch.mm(x1, x2)   
x3 = x1.mm(x2)


# Matrix exponentiation
matrix_exp = torch.rand((5, 5))
print(matrix_exp)
print(matrix_exp.matrix_power(3))  # matrix exponentiation, we dont want to elementwise exponential the matrix 
                                   # but rather we want to take the matrix and rasing the entire matrix

# elementwise multiplication
z = x * y

# Dot product
z = torch.dot(x, y)

# Batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)   # (batch, n, p)


# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations

sum_x = torch.sum(x, dim = 0) # sum along the 0th dimension
values, indices = torch.max(x, dim = 0) # returns the max value and its index
values, indices = torch.min(x, dim = 0) # returns the min value and its index
abs_x = torch.abs(x) # returns the absolute value of the tensor
z = torch.argmax(x, dim = 0) # returns the index of the max value
z = torch.argmin(x, dim = 0) # returns the index of the min value
mean_x = torch.mean(x.float(), dim = 0) # returns the mean of the tensor
z = torch.eq(x, y) # returns a tensor with the same shape as x and y and 1s where the elements are equal and 0s otherwise
sorted_y, indices = torch.sort(y, dim = 0, descending = False) # sorts the elements along the 0th dimension
z = torch.clamp(x, min = 0, max = 1) # returns a tensor with the same shape as x and y and values between 0 and 1

x = torch.tensor([1, 0, 1, 1, 1], dtype = torch.bool)
z = torch.any(x) # returns True if any element is True
z = torch.all(x) # returns True if all elements are True


# =========================================================================== #
#                          Tensor Indexing & Sliceing                         #
# =========================================================================== #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x[0,:]  get the first element of the batch

print(x[:, 0].shape)  # get the first feature over all of the examples

print(x[2, 0:10])   # 0:10 --> [0, 1, 2, .., 9] get the third row and get all the elements from 1 to 10 in that row

x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])   # pick out the 3rd, 6th, 9th example in the batch
x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])    # pick out the 2nd row and 5th column then pick out the 1st row and 1st column

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])     # pick out elements that less than 2 or greater than 8
print(x[x.remainder(2) == 0])   # pick out elements that are divisible by 2

# Useful operations
print(torch.where(x > 5, x, x * 2)) # if x > 5 then return x, else return x * 2
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()) # returns the unique elements in the tensor
print(x.ndimension()) # returns the number of dimensions of the tensor
print(x.numel()) # returns the number of elements in the tensor

# =========================================================================== #
#                             Tensor Reshaping                                #
# =========================================================================== #

x = torch.arange(9)
x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)

y = x_3x3.t() # transpose the matrix
print(y)

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim = 0)) # concatenate the two tensors along the 0th dimension
print(torch.cat((x1, x2), dim = 1)) # concatenate the two tensors along the 1st dimension

z = x1.view(-1) # -1 means that the number of elements in the dimension is not specified, 
                   # which also means flattening the entire thing

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1) # permute the tensor along the dimensions
print(z.shape)

x = torch.arange(10)    
print(x.unsqueeze(0).shape) # unsqueeze adds a dimension to the tensor
print(x.unsqueeze(1).shape) # unsqueeze adds a dimension to the tensor

x = torch.arange(10).unsqueeze(0).unsqueeze(10) # 1 x 1 x 10

z = x.squeeze(1)
print(x.shape)