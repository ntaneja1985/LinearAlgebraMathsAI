# Mathematical Foundations of Machine Learning
Linear Algebra and Mathematics for Machine Learning

## Intro to Linear Algebra
- Algebra is arithmetic that includes non-numerical entities like x:
- 2x + 5 = 25
- 2x = 20
- x = 10
- If it has an exponential term, it isnt linear algebra:
- ![alt text](image.png)
- Linear Algebra is solving for unknowns within system of linear equations 
- ![alt text](image-1.png)
- ![alt text](image-2.png)
- ![alt text](image-4.png)
- Note if the sheriff car is same speed as bank robber, there is no solution
- Also if they both start at the same time and same speed, there could be infinite solutions
- We can use the matplotlib library in python to plot this(Scott Plot in c#)
```python

import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 40, 1000) # start, finish, n points

d_r = 2.5 * t # Distance travelled by robber

d_s = 3 * (t-5) # Distance travelled by sheriff

fig, ax = plt.subplots()
plt.title('A Bank Robber Caught')
plt.xlabel('time (in minutes)')
plt.ylabel('distance (in km)')
ax.set_xlim([0, 40])
ax.set_ylim([0, 100])
ax.plot(t, d_r, c='green')
ax.plot(t, d_s, c='brown')
plt.axvline(x=30, color='purple', linestyle='--')
_ = plt.axhline(y=75, color='purple', linestyle='--')

```
- ![alt text](image-9.png)
- ![alt text](image-5.png)
- ![alt text](image-6.png)
- Here 'a' is the average house price
- The above is an example of a regression model
- ![alt text](image-7.png)
- Contemporary applications of Linear Algebra
- ![alt text](image-8.png)


## Tensors
- Tensors are basically ML generalization of vectors and matrices to any number of dimensions
- zero dimensional tensor is a scalar 
- ![alt text](image-10.png)

### Scalar Tensors
- They have no dimensions
- They are a single number
- Denoted in lowercase, italics eg: x
- Should be typed, like all other tensors: e.g int,float 
- Scalars (Rank 0 Tensors) in Base Python
```python
x = 25
x # Output is 25

type(x) # Output is int 

y=3
py_sum = x + y 
py_sum # Output is 28

type(py_sum) # Output is int

x_float = 25.0
float_sum = x_float + y 
float_sum # Output is 28.0

type(float_sum) # Output is float
```
- PyTorch and TensorFlow are the two most popular automatic differentiation libraries
### PyTorch tensors are designed to be pythonic, i.e., to feel and behave like NumPy arrays.
- The advantage of PyTorch tensors relative to NumPy arrays is that they easily be used for operations on GPU(many parallel matrix processing, useful for ML algorithms)
- We can make scalars in PyTorch also like this 
```python
import torch
x_pt = torch.tensor(25) # type specification optional, e.g.: dtype=torch.float16
x_pt # Output is tensor(25)

x_pt.shape #Output is torch.Size([])

```
### Scalars in TensorFlow can be done like this 
- Tensors created with a wrapper, all of which you can read about here:
- tf.Variable
- tf.constant
- tf.placeholder
- tf.SparseTensor
- Most widely-used is tf.Variable, which we'll use here.

As with TF tensors, in PyTorch we can similarly perform operations, and we can easily convert to and from NumPy arrays.
```python
import tensorflow as tf
x_tf = tf.Variable(25, dtype=tf.int16) # dtype is optional
x_tf # Output is <tf.Variable 'Variable:0' shape=() dtype=int16, numpy=25>

x_tf.shape # Output is TensorShape([])

y_tf = tf.Variable(3, dtype=tf.int16)
x_tf + y_tf #Output <tf.Tensor: shape=(), dtype=int16, numpy=28>

tf_sum = tf.add(x_tf, y_tf)
tf_sum #Output is <tf.Tensor: shape=(), dtype=int16, numpy=28>

tf_sum.numpy() # note that NumPy operations automatically convert tensors to NumPy arrays, and vice versa

type(tf_sum.numpy()) #Output is numpy.int16

tf_float = tf.Variable(25., dtype=tf.float16)
tf_float #Output is <tf.Variable 'Variable:0' shape=() dtype=float16, numpy=25.0>

```
### Vectors and Vector Transposition
- Vectors are one-dimensional array of numbers 
- Denoted in lowercase italics, bold e.g: **x**
- Arranged in an order so element can be accessed by its index
- Elements of a vector are scalar so not bold
- Vectors represent a particular in space 
- Vector of length 2 represents a location in a 2D Matrix
- ![alt text](image-11.png)
- Vector of length 3 represents location in a 3D cube
- Vector of length n represents location in n-dimensional tensor(difficult to imagine visually, but computers can handle it !)
- Vector Transposition-->Transforms a vector from row-vector to column-vector and vice-versa
- ![alt text](image-12.png)
- Note how the shape of the vector goes from (1,3) to (3,1)
```python
x = np.array([25, 2, 5]) # type argument is optional, e.g.: dtype=np.float16
x #Output array([25,  2,  5])

len(x) # Output is 3

x.shape # Output is (3,)

type(x) # Output is numpy.ndarray

x[0] # Output is 25 

type(x[0]) # Output is int64

# Vector Transposition
# Transposing a regular 1-D array has no effect...
x_t = x.T # Add .T in front of any vector and it will transpose it for us
x_t # Output is array([25,  2,  5])

x_t.shape # Output is (3,)

# ...but it does we use nested "matrix-style" brackets:
y = np.array([[25, 2, 5]])
y # Output is array([[25,  2,  5]])

y.shape # Output is (1,3)

# ...but can transpose a matrix with a dimension of length 1, which is mathematically equivalent:
y_t = y.T
y_t #Output is array([[25],
                    # [ 2],
                    #[ 5]])

y_t.shape # this is a column vector as it has 3 rows and 1 column
#Output is (3,1)

# Column vector can be transposed back to original row vector:
y_t.T #Output is array([[25,  2,  5]])

y_t.T.shape() # Output is (1,3)

```

### Zero Vectors
- Have no effect if added to another vector
```python
z = np.zeros(3)
z # Output is array([0., 0., 0.])

x_pt = torch.tensor([25, 2, 5])
x_pt #Output is tensor([25,  2,  5])

```

### Please note that Vectors not only represent a point in space but can also represent a magnitude and direction in space
- Remember vectors represent a point in space like this
- ![alt text](image-11.png)
- We can also think of them as representing direction and magnitude like this 
- ![alt text](image-14.png)
- Norms are a class of functions that quantify the vector magnitude(or its length)
- They describe the distance from the origin(say (0,0,0))
- Most common is L2 norm
- We take each element in our vector, square them and then add them and then take their square root 
- ![alt text](image-15.png)
- This L2 norm measures the simple (euclidean) distance from origin(looks like pythogoras theorem)
- It is the most common norm in machine learning 
- ![alt text](image-16.png)
```python
x # Output is [25,2,5]
(25**2 + 2**2 + 5**2)**(1/2) # Output is 25.573423705088842

# Get the norm
np.linalg.norm(x) #Output is np.float64(25.573423705088842)

```
- So, if units in this 3-dimensional vector space are meters, then the vector  x  has a length of 25.6m

### Unit Vectors 
- Special case where its length or L2 norm is equal to 1 
- Technically, x is a unit vector with unit norm i.e ||**x**|| = 1
- ![alt text](image-17.png)
- There are other norms also

### L1 norm
- ![alt text](image-18.png)
- Common norm in ML
- Varies linearly at all locations whether near or far from origin
- Used whenever difference between zero and non zero is the key
```python
x # Output is []
np.abs(25) + np.abs(2) + np.abs(5) #Output is 32

```
### Squared L^2 norm
- ![alt text](image-19.png)
- Here it is same as L2 norm but we dont use the square root 
- ![alt text](image-21.png)
```python
x #Output is [25,2,5]

(25**2 + 2**2 + 5**2) #Output is 654

# we'll cover tensor multiplication more soon but to prove point quickly:
np.dot(x, x) #Output is 654 (This is x * transpose(x))

```

### Max Norm(or L(infinity) norm)
- ![alt text](image-22.png)
- ![alt text](image-23.png)
```python
np.max([np.abs(25), np.abs(2), np.abs(5)]) #Output is 25

```

### Generalized Lp norm
- ![alt text](image-24.png)
- ![alt text](image-26.png)


## Basis, Orthogonal and Orthonormal Vectors
- Basis vectors can be scaled to represent any vector in a given vector space
- Typically use unit vectors along axes of vector space(shown)
- ![alt text](image-27.png)

### Orthogonal Vectors
- x and y are orthogonal vectors if x^t * y = 0
- It means vectors are 90 degree angle to each other 
- ![alt text](image-28.png)

### Orthonormal vectors 
- Orthonormal vectors are orthogonal and all have unit norm
- Basis vectors are an example
- ![alt text](image-29.png)
```python
i = np.array([1, 0])
i #Output is [1,0]

j = np.array([0, 1])
j #Output is [0,1]

np.dot(i, j) #Output is 0

```

## 2-Dimensional Tensors(also called Matrices)
- Matrix is a 2 dimensional array of numbers 
- ![alt text](image-30.png)
- Denoted in uppercase, italics and bold 
- If Matrix X has 3 rows and 2 columns, its shape is (3,2)
- Individual scalar elements are denoted in uppercase, italics only 
- ![alt text](image-31.png)
- Matrices in NumPy
- ![alt text](image-32.png)
- ![alt text](image-35.png)
- Matrices in PyTorch
- ![alt text](image-33.png)
- Matrices in TensorFlow 
- ![alt text](image-34.png)

## Generic Tensor Notation
- We can generalize our notation to be able to represent tensors with any number of dimensions, including the high dimensional tensors that are common behind the scenes in machine learning models.
- ![alt text](image-36.png)
- As an example, rank 4 tensors are common for images, where each dimension corresponds to:
- Number of images in training batch, e.g., 32
- Image height in pixels, e.g., 28 for MNIST digits
- Image width in pixels, e.g., 28
- Number of color channels, e.g., 3 for full-color images (RGB)
```python
images_pt = torch.zeros([32, 28, 28, 3]) #Create a 4D tensor with these sizes
images_pt #Output would be tensor([[[[0., 0., 0.],
          #[0., 0., 0.],
          #[0., 0., 0.],
          #...,
          #[0., 0., 0.],
          #[0., 0., 0.],
          #[0., 0., 0.]],

         #[[0., 0., 0.],
          #[0., 0., 0.],
          #[0., 0., 0.],
          #...,

```
- ![alt text](image-37.png)
- ![alt text](image-38.png)

## Common Tensor Operations
- Here we will do the following operations:
- ![alt text](image-39.png)

### Tensor Transposition
- ![alt text](image-40.png)
- ![alt text](image-41.png)

### Basic Tensor Arithmetic(including Hadamard Product)
- ![alt text](image-43.png)
- ![alt text](image-42.png)
- ![alt text](image-44.png)
- If two tensors have the same size, operations are often by default applied element-wise. This is not **matrix multiplication**, but is rather called the **Hadamard product** or simply the **element-wise product**.
- The mathematical notation is  A⊙X
- ![alt text](image-45.png)

### Tensor Reduction
- Calculating the sum across all elements of a tensor is a common operation. For example:
- ![alt text](image-46.png)
- ![alt text](image-47.png)
- We can do the sum in a 2D tensor along rows and columns also like this 
- ![alt text](image-48.png)
- Many other operations can be applied with reduction along all or a selection of axes, e.g.:
- maximum
- minimum
- mean
- product
- They're fairly straightforward and used less often than summation

## Dot Product of 2 Vectors
- ![alt text](image-49.png)
- The dot product is ubiquitous in deep learning: It is performed at every artificial neuron in a deep neural network, which may be made up of millions (or orders of magnitude more) of these neurons.
- ![alt text](image-50.png)
- Use it in PyTorch like this 
- ![alt text](image-51.png)
- Use it in TensorFlow like this
- ![alt text](image-52.png)
- ![alt text](image-53.png)
- ![alt text](image-54.png)

## Solving Linear Systems with Substitution
-  We can use matrices to solve some simple linear systems computationally i.e using code.
-  ![alt text](image-55.png)
-  ![alt text](image-56.png)
-  ![alt text](image-57.png)

## Solving Linear Systems with Elimination
- ![alt text](image-58.png)
- ![alt text](image-59.png)
- ![alt text](image-60.png)
- ![alt text](image-61.png)
```python
x = np.linspace(-10, 10, 1000) # start, finish, n points
y1 = -5 + (2*x)/3
y2 = (7-2*x)/5
fig, ax = plt.subplots()
plt.xlabel('x')
plt.ylabel('y')

# Add x and y axes:
plt.axvline(x=0, color='lightgray')
plt.axhline(y=0, color='lightgray')

ax.set_xlim([-2, 10])
ax.set_ylim([-6, 4])
ax.plot(x, y1, c='green')
ax.plot(x, y2, c='brown')
plt.axvline(x=6, color='purple', linestyle='--')
_ = plt.axhline(y=-1, color='purple', linestyle='--')

```
- ![alt text](image-62.png)


## Matrix Properties
- We will look at the following items:
- ![alt text](image-63.png)

### Frobenius Norm
- It is a function that enables us to quantify the size of a matrix.
- ![alt text](image-65.png)
- ![alt text](image-66.png)

### Matrix Multiplication
- Most widely used operation in Machine Learning
- First rule is that if we are multiplying 2 matrices, then number of columns in the first matrix should be equal to the number of rows in the second matrix.
- When we multiply the above 2 matrices, we get a new matrix where number of rows is equal to the rows of the first matrix and number of columns is equal to the number of columns in the second matrix. 
- ![alt text](image-67.png)
- ![alt text](image-69.png)
- ![alt text](image-85.png)
- ![alt text](image-70.png)
- In Pytorch we can do it like this 
- ![alt text](image-71.png)
- In Tensor flow we can do it like this 
- ![alt text](image-72.png)
- ![alt text](image-73.png)
- ![alt text](image-74.png)
- Note that AB <> BA
- ![alt text](image-75.png)
- We can do it in PyTorch as follows:
- ![alt text](image-76.png)
- Remember this algorithm where we were trying to predict the price of a house 
- ![alt text](image-77.png)
- Note that we have a set of data points to train this regression model 
- Note that every row represents a specific house.
- We can represent this in matrix form like this 
- ![alt text](image-78.png)

### Symmetry and Identity Matrices
- Symmetric is a special kind of matrix which has following properties
- It should be square
- Its transpose should be equal to the matrix itself
- ![alt text](image-79.png)
- ![alt text](image-80.png)
- Identity Matrix is a symmetric matrix where every element along diagonal is 1
- All other elements are 0
- ![alt text](image-81.png)
- ![alt text](image-82.png)
- Take an n length vector and multiply it by the identity matrix then it remains unchanged.
- ![alt text](image-83.png)
- ![alt text](image-84.png)

### Matrix Inversion
- It is clever convenient approach for solving linear equations
- An alternative to manually solving with substitution or elimination
- Matrix inverse of X is denoted as X^-1
- Multiplying a matrix by its inverse results in the identity matrix
- ![alt text](image-86.png)
- Remember the house prices example
- ![alt text](image-87.png)
- ![alt text](image-88.png)
- ![alt text](image-89.png)
- ![alt text](image-90.png)
- ![alt text](image-91.png)
- There are limitations to applying Matrix inversions 
- ![alt text](image-92.png)
- Note that columns should not be multiples of other columns
- Another problem with Matrix inversion is that it can only be calculated if the matrix is a square i.e its rows and cols are the same. 
- ![alt text](image-93.png)
- ![alt text](image-94.png)
- Note that -4,-8 is a multiple of 1,2, so its a singular matrix so we cannot calculate its inverse.
- Similarly if we try to invert a non-square matrix, then we get an error also.

### Diagonal Matrix
- Has non-zero elements along main diagonal; zeros everywhere
- Identity matrix is an example of a diagonal matrix, its a special type of diagonal matrix.
- ![alt text](image-95.png)

### Orthogonal Matrix
- Recall orthonormal vectors from earlier, i.e vectors which are perpendicular to each other. Doing their dot products results in 0. Basis vectors were an example
- ![alt text](image-96.png)
- ![alt text](image-98.png)
- ![alt text](image-99.png)
- ![alt text](image-100.png)
- Note that that the rows and columns are orthonormal, i.e they have unit norm(size is 1) and doing their dot product with other rows and columns of the matrix results in 0
- We can prove this using NumPy also 
- ![alt text](image-101.png)
- ![alt text](image-102.png)
- ![alt text](image-103.png)
- ![alt text](image-104.png)
- ![alt text](image-108.png)
- ![alt text](image-105.png)
- ![alt text](image-106.png)
- ![alt text](image-107.png)
- We've now determined that, in addition to being orthogonal, the columns of $K$ have unit norm, therefore they are orthonormal.
- To ensure that $K$ is an orthogonal matrix, we would need to show that not only does it have orthonormal columns but it has orthonormal rows are as well. Since $K^T \neq K$, we can't prove this quite as straightforwardly as we did with $I_3$.
- One approach would be to repeat the steps we used to determine that $K$ has orthogonal columns with all of the matrix's rows (please feel free to do so). Alternatively, we can use an orthogonal matrix-specific equation from the slides, $A^TA = I$, to demonstrate that $K$ is orthogonal in a single line of code
- ![alt text](image-109.png)

## Eigenvectors and EigenValues

