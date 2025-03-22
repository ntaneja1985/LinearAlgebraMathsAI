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
- Here we will use Tensors in Python to solve system of equations and identifying meaningful patterns in data
- **Determinant of a matrix** is a scalar that provides key information about how a matrix transforms other tensors.
- **Singular Value Decomposition** is used to compress data by decreasing the size of a matrix while retaining its most informative components.
- **Moore-Penrose-Pseudoinverse** is a hugely useful tool that enables us to solve for unknown values in linear systems that aren't appropriate for ordinary matrix inversion, such as the overdetermined systems of equations that are typical of machine learning.

### Review of Linear Algebra
- Solving for unknowns in a system of equations
- Remember sheriff-bank robber car example 
- ![alt text](image-110.png)
- ![alt text](image-111.png)
- ![alt text](image-112.png)
- ![alt text](image-113.png)
- ![alt text](image-114.png)
- Linear algebra can used to solve for unknowns in ML algos, including deep learning.
- We can reduce dimensionality (e.g principal component analysis)
- Linear algebra is great for ranking results(e.g with eigenvector, including in Google Page Rank algorithm)
- Linear Algebra is good for recommender systems like movie recommendations in Netflix(SVD)
- Good for NLP(e.g SVD, Matrix Factorization)
- ![alt text](image-115.png)
- ![alt text](image-116.png)
- ![alt text](image-117.png)
- ![alt text](image-118.png)
- ![alt text](image-119.png)
- ![alt text](image-120.png)
- ![alt text](image-122.png)
- ![alt text](image-123.png)
- ![alt text](image-124.png)
- Remember overdeterminations and underdeterminations
- ![alt text](image-125.png)

### Applying Matrices
- All that "Applying Matrices" means is perform Matrix Multiplication
- ![alt text](image-126.png)
- ![alt text](image-127.png)
- ![alt text](image-128.png)
- ![alt text](image-129.png)
- ![alt text](image-130.png)
- ![alt text](image-131.png)


### Affine Transformations
- These are particularly useful transformations like flipping and or rotating that we carry out by applying matrices.
- ![alt text](image-132.png)
- Let's plot  v  using my plot_vectors() function (which is based on Hadrien Jean's plotVectors() function
```python
v = np.array([3, 1])
v # Output is array([3, 1])

import matplotlib.pyplot as plt
def plot_vectors(vectors, colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each. 

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]] 
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
        
    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0,0],vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=colors[i],)


# Run the above program like this
plot_vectors([v], ['lightblue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

```
- Result is the following:
- ![alt text](image-133.png)
- "Applying" a matrix to a vector (i.e., performing matrix-vector multiplication) can linearly transform the vector, e.g, rotate it or rescale it.
- The identity matrix, introduced earlier, is the exception that proves the rule: Applying an identity matrix does not transform the vector as shown below:

```python

v = np.array([3, 1])
v # Output is array([3, 1])

I = np.array([[1, 0], [0, 1]])
I # Output is array([[1, 0],
                    #[0, 1]])
Iv = np.dot(I, v)
Iv # Output is array([3, 1])

plot_vectors([Iv], ['blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

```
- Result is the same as earlier when we just plotted the vector in light-blue color
- ![alt text](image-134.png)

- In contrast, consider this non-identity matrix (let's call it  E ) that flips vectors over the  x -axis:

```python

v = np.array([3, 1])
v # Output is array([3, 1])

E = np.array([[1, 0], [0, -1]])
E # Output is array([[ 1,  0],
                    #[ 0, -1]])

Ev = np.dot(E, v)
Ev #Output is array([ 3, -1])

plot_vectors([v, Ev], ['lightblue', 'blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-3, 3)

```
- Result is ![alt text](image-135.png)

- Lets consider another matrix F, that flips vectors over the y-axis
```python
F = np.array([[-1, 0], [0, 1]])
F # Output is array([[-1,  0],
                    #[ 0,  1]])
Fv = np.dot(F, v)
Fv #Output is array([-3,  1])

plot_vectors([v, Fv], ['lightblue', 'blue'])
plt.xlim(-4, 4)
_ = plt.ylim(-1, 5)
```
- Result is 
- ![alt text](image-136.png)
- Applying a flipping matrix is an example of an **affine transformation**: a change in geometry that may adjust distances or angles between vectors, but preserves parallelism between them.
- In addition to flipping a matrix over an axis (a.k.a., reflection), other common affine transformations include:
- Scaling (changing the length of vectors)
- Shearing (example of this on the Mona Lisa coming up shortly)
- Rotation
- A single matrix can apply multiple affine transforms simultaneously (e.g., flip over an axis and rotate 45 degrees). As an example, let's see what happens when we apply this matrix  A  to the vector  v :
  
```python
v = np.array([3, 1])
v # Output is array([3, 1])

A = np.array([[-1, 4], [2, -2]])
A #Output is array([[-1,  4],
                   #[ 2, -2]])

Av = np.dot(A, v)
Av #Output is array([1, 4])

plot_vectors([v, Av], ['lightblue', 'blue'])
plt.xlim(-1, 5)
_ = plt.ylim(-1, 5)

```
- Result is ![alt text](image-137.png)
- We can concatenate several vectors together into a matrix (say, $V$), where each column is a separate vector. Then, whatever linear transformations we apply to $V$ will be independently applied to each column (vector): 
```python
v = np.array([3, 1])
v # Output is array([3, 1])
v2 = np.array([2, 1])

# recall that we need to convert array to 2D to transpose into column, e.g.:
np.matrix(v).T 

v3 = np.array([-3, -1]) # mirror image of v over both axes
v4 = np.array([-1, 1])

V = np.concatenate((np.matrix(v).T, 
                    np.matrix(v2).T,
                    np.matrix(v3).T,
                    np.matrix(v4).T), 
                   axis=1)
V #Output is matrix([[ 3,  2, -3, -1],
                    #[ 1,  1, -1,  1]])

IV = np.dot(I, V)
IV #Output is matrix([[ 3,  2, -3, -1],
                    # [ 1,  1, -1,  1]])

A = np.array([[-1, 4], [2, -2]])
A #Output is array([[-1,  4],
                   #[ 2, -2]])
AV = np.dot(A, V)
AV #Output is matrix([[ 1,  2, -1,  5],
                     #[ 4,  2, -4, -4]])

# function to convert column of matrix to 1D vector: 
def vectorfy(mtrx, clmn):
    return np.array(mtrx[:,clmn]).reshape(-1)

vectorfy(V, 0) #Output is array([3, 1])

plot_vectors([vectorfy(V, 0), vectorfy(V, 1), vectorfy(V, 2), vectorfy(V, 3),
             vectorfy(AV, 0), vectorfy(AV, 1), vectorfy(AV, 2), vectorfy(AV, 3)], 
            ['lightblue', 'lightgreen', 'lightgray', 'orange',
             'blue', 'green', 'gray', 'red'])
plt.xlim(-4, 6)
_ = plt.ylim(-5, 5)

```
- Result is ![alt text](image-138.png)
- Affine transformations are linear transformations followed by translation. They are useful because they allow geometric manipulations while maintaining the basic structure of objects. For example, in your pharmacy management system, affine transformations might be useful in data visualization when adjusting graphs or plots to different scales or orientations. 

## Eigenvectors and Eigenvalues
- ![alt text](image-139.png)
- ![alt text](image-140.png)
- Being on same span means being on the same line
- Note in shearing matrix, the Red vector is knocked off its span, but blue vector is not. 
- So blue vector is an eigenvector but not the red vector.
- **Eigenvalues** are scalar values that tell you how much the eigenvectors length has changed as a result of applying the particular matrix that we're applying.
- ![alt text](image-141.png)
- If eigenvector were to double in length, its eigenvalue = 2; if it halves, eigenvalue = 0.5, if we fip and shear, then eigenvalue = -1
- ![alt text](image-142.png)
- Similarly if the eigenvector were to double in length, while reversing direction, eigenvalue would be -2
- An **eigenvector** (*eigen* is German for "typical"; we could translate *eigenvector* to "characteristic vector") is a special vector $v$ such that when it is transformed by some matrix (let's say $A$), the product $Av$ has the exact same direction as $v$.
- An **eigenvalue** is a scalar (traditionally represented as $\lambda$) that simply scales the eigenvector $v$ such that the following equation is satisfied: 
- $Av = \lambda v$
- This is rather confusing, on the left-side we are applying a matrix to a vector and on the right-side we are applying a scalar to a vector
- Eigenvectors and eigenvalues can be derived algebraically (e.g., with the QR algorithm, which was independently developed in the 1950s by both Vera Kublanovskaya and John Francis), however this is outside scope of the ML Foundations series. We'll cheat with NumPy eig() method, which returns a tuple of:
- a vector of eigenvalues
- a matrix of eigenvectors
- Essentially, eigenvectors and eigenvalues act as magnifying glasses: they help you focus on the most meaningful directions and magnitudes of change in your data or systems. It's like having a secret tool that cuts through complexity so you can make smarter decisions.
- ![alt text](image-145.png)
- ![alt text](image-146.png)
```python
A = np.array([[-1, 4], [2, -2]])
A #Output is array([[-1,  4],
                   #[ 2, -2]])


lambdas, V = np.linalg.eig(A) 
# The matrix contains as many eigenvectors as there are columns of A:
V #Output is array([[ 0.86011126, -0.76454754],
                   #[ 0.51010647,  0.64456735]])

# With a corresponding eigenvalue for each eigenvector:
lambdas #Output is array([ 1.37228132, -4.37228132])

# Let's confirm that  Av=λv  for the first eigenvector:

v = V[:,0] 
v # Output is array([0.86011126, 0.51010647])

lambduh = lambdas[0] # note that "lambda" is reserved term in Python
lambduh #Output is np.float64(1.3722813232690143)

Av = np.dot(A, v)
Av #Output is array([1.18031462, 0.70000958])

lambduh * v #Output is array([1.18031462, 0.70000958])

plot_vectors([Av, v], ['blue', 'lightblue'])
plt.xlim(-1, 2)
_ = plt.ylim(-1, 2)


# again for the second eigenvector of A:
v2 = V[:,1]
v2 #Output is array([-0.76454754,  0.64456735])

lambda2 = lambdas[1]
lambda2 #Output is -4.372281323269014

Av2 = np.dot(A, v2)
Av2 #Output is array([ 3.34281692, -2.81822977])

lambda2 * v2 #Output is array([ 3.34281692, -2.81822977])

plot_vectors([Av, v, Av2, v2], 
            ['blue', 'lightblue', 'green', 'lightgreen'])
plt.xlim(-1, 4)
_ = plt.ylim(-3, 2)

```
- Result for the first one is ![alt text](image-143.png)
- Result for the second plot is ![alt text](image-144.png)

### Eigenvectors in greater than 2 dimensions
- While plotting gets trickier in higher-dimensional spaces, we can nevertheless find and use eigenvectors with more than two dimensions. Here's a 3D example (there are three dimensions handled over three rows):
```python
X = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])
X #Output is array([[25,  2,  9],
                   #[ 5, 26, -5],
                   #[ 3,  7, -1]])
lambdas_X, V_X = np.linalg.eig(X)
V_X #Output is array([[-0.71175736, -0.6501921 , -0.34220476],
                     #[-0.66652125,  0.74464056,  0.23789717],
                     #[-0.22170001,  0.15086635,  0.90901091]])

lambdas_X # a corresponding eigenvalue for each eigenvector
    #Output is array([29.67623202, 20.62117365, -0.29740567])

# Confirm  Xv=λv  for an example eigenvector:
v_X = V_X[:,1] 
v_X #Output is array([-0.6501921 ,  0.74464056,  0.15086635])

lambda_X = lambdas_X[1] 
lambda_X #Output is np.float64(20.62117365053535)

np.dot(X, v_X) # matrix multiplication
# Output is array([-13.40772428,  15.3553624 ,   3.11104129])

lambda_X * v_X
#Output is array([-13.40772428,  15.3553624 ,   3.11104129])
# As you can see Xv = λv

```

### Matrix Determinants
- A determinant is a special scalar value that we can calculate for any given matrix.
- It has a number of very useful properties as well as an intimate relationship with eigenvalues
- Matrix determinants map a square matrix to a single special scalar value.
- The key here is that we do need to have square matrices in order to calculate their determinant.
- It enables us to determine whether a matrix can be inverted.
- ![alt text](image-147.png)
- ![alt text](image-148.png)
```python
X = np.array([[4, 2], [-5, -3]])
X # Output is array([[ 4,  2],
                    #[-5, -3]])

np.linalg.det(X) #Output is -2
```
- ![alt text](image-149.png)
- As determinant is 0, no solution is possible as it is a singular matrix
- ![alt text](image-150.png)

### Determinants of Larger Matrices
- We will use recursion
- Lets say we have a 5 x 5 matrix
- We will first calculate determinant of bottom 2 x 2 matrix, then 3 x 3 matrix, then 4 x 4 matrix and finally 5 x 5 matrix
- ![alt text](image-151.png)
- ![alt text](image-152.png)
```python
X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
X #Output is array([[ 1,  2,  4],
                   #[ 2, -1,  3],
                   #[ 0,  5,  1]])
np.linalg.det(X) #Output is 20

```
### Determinants and EigenValues
- det(X) = product of all eigenvalues of X (lambdas)
```python
X = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])
X #Output is array([[ 1,  2,  4],
                   #[ 2, -1,  3],
                   #[ 0,  5,  1]])
np.linalg.det(X) #Output is 20


lambdas, V = np.linalg.eig(X)
lambdas #EigenValues
#Output is array([-3.25599251, -1.13863631,  5.39462882])

np.prod(lambdas) #Output is 20 (same as the determinant)

```
- |det(X)| quantifies volume change as a result of applying X to some tensor
- ![alt text](image-153.png)
- If any one of a matrix's eigenvalues is zero, then the product of the eigenvalues must be zero and the determinant must also be zero.
- The determinant represents how a matrix scales volume during its transformation.
- Eigenvalues describe scaling factors along their corresponding eigenvectors. Multiplying all eigenvalues gives the total scaling effect, which matches the determinant.
- The determinant is used in assessing properties of eigenvalues, which are vital for dimensionality reduction techniques like PCA.
- In systems like control theory or reinforcement learning, the determinant (product of eigenvalues) helps analyze whether a system is stable.
- In systems like control theory or reinforcement learning, the determinant (product of eigenvalues) helps analyze whether a system is stable.

### Eigen Decomposition
- Described by this formula
- ![alt text](image-154.png)
- ![alt text](image-155.png)
- The **eigendecomposition** of some matrix $A$ is 

$A = V \Lambda V^{-1}$

Where: 

* As in examples above, $V$ is the concatenation of all the eigenvectors of $A$
* $\Lambda$ (upper-case $\lambda$) is the diagonal matrix diag($\lambda$). Note that the convention is to arrange the lambda values in descending order; as a result, the first eigenvalue (and its associated eigenvector) may be a primary characteristic of the matrix $A$.

```python
# This was used earlier as a matrix X; it has nice clean integer eigenvalues...
A = np.array([[4, 2], [-5, -3]]) 
A #Output is array([[ 4,  2],
                   #[-5, -3]])
lambdas, V = np.linalg.eig(A)
V #Output is eigenvectors array([[ 0.70710678, -0.37139068],
                   #[-0.70710678,  0.92847669]])
lambdas # Output is array([ 2., -1.])
Vinv = np.linalg.inv(V)
Vinv #Output is array([[2.3570226 , 0.94280904],
                      #[1.79505494, 1.79505494]])
Lambda = np.diag(lambdas)
Lambda #Output is array([[ 2.,  0.],
                        #[ 0., -1.]])

np.dot(V, np.dot(Lambda, Vinv)) #Output is array([[ 4,  2],
                                                 #[-5, -3]])

```
- The above confirms that $A = V \Lambda V^{-1}$: 
- Eigendecomposition is not possible with all matrices. And in some cases where it is possible, the eigendecomposition involves complex numbers instead of straightforward real numbers.
- In machine learning, however, we are typically working with real symmetric matrices, which can be conveniently and efficiently decomposed into real-only eigenvectors and real-only eigenvalues.
- If $A$ is a real symmetric matrix then...

$A = Q \Lambda Q^T$

...where $Q$ is analogous to $V$ from the previous equation except that it's special because it's an orthogonal matrix. 
 - Remember it is cheap to compute the transpose of a matrix compared to its inverse. 
 - Remember Q^T * Q = I (Identity Matrix)

### EigenVector and EigenValue Applications
- Eigenvectors and eigenvalues reveal the core properties of a matrix:
- What it does (directions and scaling),
- How it acts (stability, dimensionality),
- And how to simplify or analyze its effects
- ![alt text](image-156.png)
- ![alt text](image-158.png)
- ![alt text](image-159.png)
- Singular value decomposition is used to compress the size of media files of data files.
- Moore-penrose Pseudoinverse is used to fit a regression line to points 
- Principal Component Analysis is used to identify some underlying structure in unlabeled data.

## Matrix Operations for Machine Learning