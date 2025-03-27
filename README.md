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
```python
P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])

eValuesP,eVectorsP = torch.linalg.eig(P)
      
singular1 = torch.diag(eValuesP)                               
singular1
inverseVectors = torch.inverse(eVectorsP)
inverseVectors
result1 = torch.mm(eVectorsP, torch.mm(singular1, inverseVectors))
result1

```

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
- ![alt text](image-160.png)

### Singular Value Decomposition(SVD)
- Unlike eigen decomposition, which is applicable to square matrices only, Singular value decomposition or SVD is applicable to any real valued matrix.
- ![alt text](image-161.png)
- SVD of matrix $A$ is: 

$A = UDV^T$

Where: 

* $U$ is an orthogonal $m \times m$ matrix; its columns are the **left-singular vectors** of $A$.
* $V$ is an orthogonal $n \times n$ matrix; its columns are the **right-singular vectors** of $A$.
* $D$ is a diagonal $m \times n$ matrix; elements along its diagonal are the **singular values** of $A$.

```python
A = np.array([[-1, 2], [3, -2], [5, 7]])
A #Output is array([[-1,  2],
                   #[ 3, -2],
                   #[ 5,  7]])

U, d, VT = np.linalg.svd(A) # V is already transposed

U #Output is array([[ 0.12708324,  0.47409506,  0.87125411],
                  # [ 0.00164602, -0.87847553,  0.47778451],
                  # [ 0.99189069, -0.0592843 , -0.11241989]])

VT #Output is array([[ 0.55798885,  0.82984845],
                    #[-0.82984845,  0.55798885]])

d  #Output is array([8.66918448, 4.10429538])

np.diag(d) #Output is array([[8.66918448, 0.        ],
                            #[0.        , 4.10429538]])

# D  must have the same dimensions as A for UDVT matrix multiplication to be possible:

D = np.concatenate((np.diag(d), [[0, 0]]), axis=0)
D #Output is array([[8.66918448, 0.        ],
                   #[0.        , 4.10429538],
                   #[0.        , 0.        ]])

np.dot(U, np.dot(D, VT)) #Output is array([[-1.,  2.],
                                          #[ 3., -2.],
                                          #[ 5.,  7.]])

```
- SVD and eigendecomposition are closely related to each other: 

* Left-singular vectors of $A$ = eigenvectors of $AA^T$.
* Right-singular vectors of $A$ = eigenvectors of $A^TA$.
* Non-zero singular values of $A$ = square roots of eigenvalues of $AA^T$ = square roots of eigenvalues of $A^TA$

### Data Compression with SVD
- Singular Value Decomposition (SVD) is a powerful tool from linear algebra that can be used to compress images efficiently by reducing the amount of data needed to represent them while preserving essential features.
- SVD decomposes a matrix A (representing the image) into three matrices:
- 𝐴 = 𝑈Σ𝑉^𝑇

- U is an m×m orthogonal matrix whose columns are the left singular vectors.

- Σ is an m×n diagonal matrix containing the singular values (non-negative, in descending order).
- 𝑉^𝑇 is an n×n orthogonal matrix whose rows are the right singular vectors.
- For an image, A is typically an m×n matrix where each entry represents a pixel's intensity (in grayscale) or a channel value (e.g., in RGB images).

```python
from PIL import Image
wget https://raw.githubusercontent.com/jonkrohn/DLTFpT/master/notebooks/oboe-with-book.jpg
img = Image.open('oboe-with-book.jpg')
_ = plt.imshow(img)

# Convert image to grayscale so that we don't have to deal with the complexity of multiple color channels:
imggray = img.convert('LA')
_ = plt.imshow(imggray)


#Convert data into numpy matrix, which doesn't impact image data:

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
_ = plt.imshow(imgmat, cmap='gray')

# Calculate SVD of the image
U, sigma, V = np.linalg.svd(imgmat)

# As eigenvalues are arranged in descending order in diag( λ ) so too are singular values, by convention, arranged in descending order in  D  (or, in this code, diag( σ )). Thus, the first left-singular vector of  U  and first right-singular vector of  V  may represent the most prominent feature of the image:

reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
_ = plt.imshow(reconstimg, cmap='gray')

# Additional singular vectors improve the image quality:
for i in [2, 4, 8, 16, 32, 64]:
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()

# With 64 singular vectors, the image is reconstructed quite well, however the data footprint is much smaller than the original image:

```
- ![alt text](image-162.png)
- Specifically, the image represented as 64 singular vectors is 3.7% of the size of the original!
- Alongside images, we can use singular vectors for dramatic, lossy compression of other types of media files.

### Moore-Penrose Pseudoinverse
- It can almost magically! solve for unknowns in a system of linear equations
- In linear algebra, the inversion of a matrix refers to finding a matrix that, when multiplied with the original matrix, yields the identity matrix. This "inverse" matrix essentially "undoes" the effect of the original matrix in a linear transformation.
- Determinant of this matrix should also be non-zero
- Remember if det(A) is zero, the matrix is a singular matrix
- ![alt text](image-163.png)
- ![alt text](image-164.png)
- Moore-Penrose PseudoInverse helps us to invert non-square matrices. 
- ![alt text](image-165.png)
- However solving an equation maybe still be possible by other means even if matrix cannot be inverted
- One such technique is Moore-Penrose PseudoInverse 
- ![alt text](image-166.png)
```python
A #Output is array([[-1,  2],
                   #[ 3, -2],
                   #[ 5,  7]])

# U = left singular vectors of A
# V = right singular vectors of A
# d = eigenValues/singular values
U, d, VT = np.linalg.svd(A)       

#To create  D+ , we first invert the non-zero values of  d :
D = np.diag(d)
D #Output is array([[8.66918448, 0.        ],
                  #[0.        , 4.10429538]])

# Remember Dplus  = Transpose of D with reciprocal of all non-zero elements
# 1/8.669 = 0.11535355865728457
# 1/4.104 = 0.24366471734892786

Dinv = np.linalg.inv(D)
Dinv #Output is array([[0.1153511 , 0.        ],
                     #[0.        , 0.24364718]])

# D+  must have the same dimensions as AT in order for VD+UT matrix multiplication to be possible

Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)
Dplus #Output is array([[0.1153511 , 0.        , 0.        ],
                      #[0.        , 0.24364718, 0.        ]])

# Now we have everything we need to calculate  A+  with  V * D^+ * U^T :

np.dot(VT.T, np.dot(Dplus, U.T)) 
#Output is array([[-0.08767773,  0.17772512,  0.07582938],
                 #[ 0.07661927, -0.1192733 ,  0.08688784]])

# Working out this derivation is helpful for understanding how Moore-Penrose pseudoinverses work, but unsurprisingly NumPy is loaded with an existing method pinv()
np.linalg.pinv(A)
#Output is array([[-0.08767773,  0.17772512,  0.07582938],
                 #[ 0.07661927, -0.1192733 ,  0.08688784]])
```

### Regression with PseudoInverse 
- ![alt text](image-167.png)
- It would be difficult to find as many houses in our dataset as we have features
- So Obviously it is not a square matrix and it is not easily invertable.
- ![alt text](image-168.png)
- ![alt text](image-169.png)
- ![alt text](image-170.png)
- overdetermined system means number of rows is greater than number of columns
- underdetermined system means number of rows is less than number of columns
- ![alt text](image-171.png)
- In Deep Learning models, possibility of underdetermined system is high
- For regression problems, we typically have many more cases ( n , or rows of  X ) than features to predict (columns of  X ). Let's solve a miniature example of such an overdetermined situation.
- We have eight data points ( n  = 8):
```python
x1 = [0, 1, 2, 3, 4, 5, 6, 7.] # E.g.: Dosage of drug for treating Alzheimer's disease
y = [1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37] # E.g.: Patient's "forgetfulness score"

title = 'Clinical Trial'
xlabel = 'Drug dosage (mL)'
ylabel = 'Forgetfulness'


fig, ax = plt.subplots()
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
_ = ax.scatter(x1, y)


```
- ![alt text](image-172.png)
- Although it appears there is only one predictor ( x1 ), our model requires a second one (let's call it  x0 ) in order to allow for a  y -intercept. Without this second variable, the line we fit to the plot would need to pass through the origin (0, 0). The  y -intercept is constant across all the points so we can set it equal to 1 across the board:
```python
x0 = np.ones(8)
x0 #Output is array([1., 1., 1., 1., 1., 1., 1., 1.])

# Concatenate  x0  and  x1  into a matrix  X :
X = np.concatenate((np.matrix(x0).T, np.matrix(x1).T), axis=1)
X #Output is matrix([[1., 0.],
                    #[1., 1.],
                    #[1., 2.],
                    #[1., 3.],
                    #[1., 4.],
                    #[1., 5.],
                    #[1., 6.],
                    #[1., 7.]])

w = np.dot(np.linalg.pinv(X), y)
w #Output is matrix([[ 1.76      , -0.46928571]])

#The first weight corresponds to the  y -intercept of the line, which is typically denoted as  b :
b = np.asarray(w).reshape(-1)[0]
b
 #Output is np.float64(1.7599999999999985)

#While the second weight corresponds to the slope of the line, which is typically #denoted as  m :
m = np.asarray(w).reshape(-1)[1]
m
#Output is np.float64(-0.4692857142857139)

fig, ax = plt.subplots()

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

ax.scatter(x1, y)

x_min, x_max = ax.get_xlim()
y_at_xmin = m*x_min + b
y_at_xmax = m*x_max + b

ax.set_xlim([x_min, x_max])
_ = ax.plot([x_min, x_max], [y_at_xmin, y_at_xmax], c='C01')


```
- From the slides, we know that we can calculate the weights $w$ using the equation $w = X^+y$: 
- ![alt text](image-173.png)

### Trace Operator
- It frequently comes in handy for rearranging linear algebra equations, including ones that are common in machine learning.
- Denoted as Tr($A$). Simply the sum of the diagonal elements of a matrix: $$\sum_i A_{i,i}$$
- The trace operator has a number of useful properties that come in handy while rearranging linear algebra equations, e.g.:

* Tr($A$) = Tr($A^T$)
* Assuming the matrix shapes line up: Tr($ABC$) = Tr($CAB$) = Tr($BCA$)

- In particular, the trace operator can provide a convenient way to calculate a matrix's Frobenius norm: $$||A||_F = \sqrt{\mathrm{Tr}(AA^\mathrm{T})}$$

```python
A = np.array([[25, 2], [5, 4]])
A #Output is array([[25,  2],
                   #[ 5,  4]])
25 + 4 #Output is 29

np.trace(A) #Output is 29


#Another example
A_p #Output is tensor([[-1.,  2.],
                      #[ 3., -2.],
                      #[ 5.,  7.]])
frobenius_norm_p = torch.sqrt(torch.trace(torch.matmul(A_p,A_p.T)))
frobenius_norm_p #Output is tensor(9.5917)
norm2 = torch.norm(A_p) #Calculate the frobenius norm
norm2 == frobenius_norm_p #Output is true, hence proved
```

### Principal Component Analysis
- It is a prevalent and powerful machine learning technique for working with unlabeled data.
- Principal component analysis is a simple machine learning algorithm.
- It is unsupervised, so this means that it enables the identification of structure in unlabeled data.
- If you have a bunch of data, say you have a bunch of measurements of flowers, but you don't have any labels for what kinds of flowers they are, you can nevertheless take those measurements and use an unsupervised learning algorithm like PCA to identify underlying structure in your data.
- ![alt text](image-174.png)
```python
from sklearn import datasets
iris = datasets.load_iris() ##load a sample dataset using the scikitlearn library

iris.data.shape #Output is (150,4) meaning there are 150 flowers with 4 features specified for each

iris.get("feature_names") #Output is the name of features for each of the flowers
                          # ['sepal length (cm)',
                            #'sepal width (cm)',
                            #'petal length (cm)',
                             #'petal width (cm)']

# Take the first 6 flowers
iris.data[0:6,:] #Output is array([[5.1, 3.5, 1.4, 0.2],
                                  #[4.9, 3. , 1.4, 0.2],
                                  #[4.7, 3.2, 1.3, 0.2],
                                  #[4.6, 3.1, 1.5, 0.2],
                                  #[5. , 3.6, 1.4, 0.2],
                                  #[5.4, 3.9, 1.7, 0.4]])

from sklearn.decomposition import PCA
pca = PCA(n_components=2) #Return 2 principal components that account for the most structure

# Using the flower data, return the 2 principal components for each flower
X = pca.fit_transform(iris.data)

X.shape #Output is (150,2)

X[0:6,:] #Note here for the first 6 flowers, we are showing the 2 principal **components**
# Output is array([[-2.68412563,  0.31939725],
                  #[-2.71414169, -0.17700123],
                  #[-2.88899057, -0.14494943],
                  #[-2.74534286, -0.31829898],
                  #[-2.72871654,  0.32675451],
                  #[-2.28085963,  0.74133045]])

_ = plt.scatter(X[:, 0], X[:, 1])

# Fortunately we have some labels to understand the data
# We will use it just to colorize the scatter plots
iris.target.shape #Output is (150,)
iris.target[0:6] #Output is array([0, 0, 0, 0, 0, 0])

unique_elements, counts_elements = np.unique(iris.target, return_counts=True)

# The following lines shows that for 150 flowers dataset, each of 50 flowers are different i.e they are different flowers as indicated by their labels: 0,1,2
np.asarray((unique_elements, counts_elements)) #Output is array([[ 0,  1,  2],
                                                                #[50, 50, 50]])

# Output the 3 different types of flowers some names
list(iris.target_names)
#Output is [np.str_('setosa'), np.str_('versicolor'), np.str_('virginica')]

#Lets plot them
_ = plt.scatter(X[:, 0], X[:, 1], c=iris.target)

```
- ![alt text](image-175.png)
- ![alt text](image-176.png)
- As we can see now we can observe that different flowers occupy different section of the plot.
- If we now have to predict any flower, we just need to calculate their euclidean distance from the closest plot points and then we can actually make an accurate prediction. 
- Using principal component analysis we can breakdown a large dataset into its most unique components which give most meaningful information about the data.
- Internally it uses eigenvectors and eigenvalues only
- ![alt text](image-177.png) 
- ![alt text](image-178.png)

### Final thoughts on Linear Algebra
- We talked about singular value decomposition, which allowed us to compress an image file.
- We talked about the Moore-penrose Pseudoinverse, which allowed us to perform something like matrix inversion of non-square matrices, enabling us to solve for unknowns in systems of equations like those that are common in machine learning.
- We learned about the trace operator quickly, and then we learned about principal component analysis, which ties together the trace operator and lots of other concepts that we learned earlier in this Machine Learning Foundation series to Power PCA, which is a simple machine learning algorithm for handling unlabeled data and finding structure in it.

# Calculus in Machine Learning

## Limits 
- ![alt text](image-179.png)

### Intro to Differential Calculus
- Calculus is the mathematical study of continuous change
- 2 main branches
- Differential Calculus
- Integral Calculus

### Differential Calculus
- Study of the rate of change
- ![alt text](image-180.png)
- ![alt text](image-182.png)
- ![alt text](image-183.png) (Here curve is gradually increasing in steepness)
- We take a look at the tangents
- ![alt text](image-184.png)
- When vehicle is stationary, slope is zero
- ![alt text](image-185.png)
- When vehicle is travelling at constant speed
- ![alt text](image-186.png)
- ![alt text](image-187.png)
- We can calculate these slopes using differential calculus
- ![alt text](image-188.png)
- Common to denote slopes as m
- ![alt text](image-189.png)
- ![alt text](image-191.png)
- ![alt text](image-192.png)

### Integral Calculus
- Study of area under curves
- Facilitates the inverse of differential calculus
- ![alt text](image-193.png)
- Note that AUC = Total distance (d) travelled

### Method of Exhaustion
- Allows us to identify the area of shapes
- ![alt text](image-194.png)
- ![alt text](image-196.png)
### Calculus of the Infinitesimals
- ![alt text](image-197.png)

### Applications of Calculus
- ![alt text](image-198.png)
- In Deep learning algorithms we need to find the gradient descent
- Imagine trying to train a machine learning model, like predicting house prices or recognizing faces. The model needs to adjust its parameters (weights) to minimize error—just like the hiker finding the lowest point of the mountain. Gradient descent helps the model figure out how to tweak its parameters step by step to improve accuracy.
- Think of learning a new skill—say playing piano. At first, you make random mistakes (you’re at a "high error" point). After each practice session, you figure out which mistakes to fix and adjust step by step (heading downhill on the error mountain). Eventually, after enough practice, you hit a point where you’re playing smoothly (the bottom of the valley).
- ![alt text](image-199.png)
- As we can see above when we train a ML model, the cost keeps decreasing and its accuracy keeps increasing
- We can also have Gradient ascent to maximize reward(Q-Learning)
- Higher-order derivatives used in "fancy" optimizers. Remember distance vs time to speed vs time to acceleration over gravity over time.
- Receiver Operating Characteristic used in Binary Classification
- ![alt text](image-200.png)

### Calculating Limits 
- ![alt text](image-201.png)
- ![alt text](image-202.png)
- ![alt text](image-203.png)
- ![alt text](image-204.png)
- ![alt text](image-205.png)
- ![alt text](image-206.png)

## Derivatives and Differentiation

### The Delta Method
- ![alt text](image-208.png)
- ![alt text](image-209.png)
- ![alt text](image-211.png)
- ![alt text](image-210.png)
- ![alt text](image-213.png)
- ![alt text](image-214.png)
- ![alt text](image-215.png)
- ![alt text](image-216.png)

### Derivative Notation
- ![alt text](image-217.png)

### Derivative of a constant 
- ![alt text](image-218.png)


### Power Rule
- ![alt text](image-219.png)

### Constant Multiple Rule
- ![alt text](image-220.png)
- ![alt text](image-221.png)

### The Sum Rule
- ![alt text](image-222.png)
- ![alt text](image-223.png)
- ![alt text](image-224.png)
- ![alt text](image-225.png)
- ![alt text](image-226.png)

### The Product Rule
- ![alt text](image-227.png)

### The Quotient Rule
- ![alt text](image-228.png)


### The Chain Rule
- The chain rule has many applications within machine learning.
- It is used for gradient descent
- Gradient descent is found in a huge number of machine learning algorithms from simple regression models all the way through to the most sophisticated deep learning models.
- ![alt text](image-229.png)
- ![alt text](image-230.png)
- ![alt text](image-231.png)
- ![alt text](image-232.png)
- ![alt text](image-233.png)
- ![alt text](image-234.png)
- ![alt text](image-235.png)
- ![alt text](image-236.png)

### Power Rule on a Function Chain
- ![alt text](image-237.png)


## Automatic Differentiation
- It is a computational technique that allows us to scale up the calculation of derivatives to the massive function chains that are common in machine learning.
- ![alt text](image-238.png)
- Chain rule starts differentiation from the inner-most function onward, the autodiff proceeds from the outermost function inward
- ![alt text](image-239.png)

### Autodiff with PyTorch
  **TensorFlow** and **PyTorch** are the two most popular automatic differentiation libraries.
Let's use them to calculate $dy/dx$ at $x = 5$ where: 
$$y = x^2$$
$$ \frac{dy}{dx} = 2x = 2(5) = 10 $$
```python
import torch

x = torch.tensor(5.0)

x #Output is tensor(5.)

x.requires_grad_() # contagiously track gradients through forward pass
#Output is tensor(5., requires_grad=True)

y = x**2

y.backward() # use autodiff

x.grad #Output is tensor(10.)



```
- ![alt text](image-240.png)


## Line Equation as a Tensor Graph
- ![alt text](image-241.png)
- Note that the tensors(edges) hold some information
- Autodiff, or automatic differentiation, is a game-changer in machine learning because it sidesteps a lot of the messiness that comes with solving equations the old-school algebraic way.
- First off, autodiff computes derivatives numerically with pinpoint accuracy, down to machine precision, by breaking down complex functions into their basic building blocks and applying the chain rule step-by-step. Algebraic differentiation, on the other hand, requires you to manually work out the derivative of a function symbolically—like solving a math puzzle by hand.
- That’s fine for something simple like f(x) = x², where the derivative is just 2x, but in machine learning, you’re dealing with insane, nested, multivariable monstrosities—think neural networks with millions of parameters. Trying to derive those by hand is a nightmare, prone to human error, and scales about as well as a paper umbrella in a hurricane.
- Second, autodiff is automatic. You feed it a function, and it spits out the gradient without you having to think about the math. Tools like PyTorch or TensorFlow have it built in—you write the forward pass, and they handle the backward pass (i.e., the gradients) for free. With the algebraic approach, you’d need to derive every partial derivative yourself, which isn’t just tedious but also impractical when your model changes. Tweak a neural network layer? Good luck rewriting all those equations. Autodiff doesn’t care—it adapts instantly.
- Third, it’s computationally efficient. Autodiff comes in two flavors: forward mode and reverse mode. Reverse mode (aka backpropagation) is especially clutch for machine learning because it computes gradients for all parameters in a single pass, no matter how many inputs or outputs you’ve got. Algebraic methods don’t have that kind of streamlined elegance—you’d be solving a sprawling system of equations, often redundantly, and likely approximating anyway if it gets too hairy.
- Let’s start with a basic function, f(x) = x² + 3x + 1. Algebraically, the derivative is df/dx = 2x + 3. With autodiff, we don’t need to figure that out manually—PyTorch does it for us.
```python
import torch

# Define the input (requires_grad=True tells PyTorch to track gradients)
x = torch.tensor(2.0, requires_grad=True)

# Define the function
f = x**2 + 3*x + 1

# Compute the derivative (backward pass)
f.backward()

# The gradient is stored in x.grad
print(f"Function value: {f.item()}")
print(f"Gradient (df/dx): {x.grad.item()}")

#Outputs
# Function value: 11.0
# Gradient (df/dx): 7.0
```
- What’s happening here? At x = 2, the function evaluates to 2² + 3*2 + 1 = 11, and the derivative 2*2 + 3 = 7. PyTorch’s autodiff tracked the operations and applied the chain rule automatically. No pencil-and-paper required. Imagine doing this for f(x) = sin(x) * exp(x²) + log(x)—algebraic derivation gets ugly fast, but autodiff doesn’t break a sweat.
- Now let’s scale it up to a tiny neural network with one layer. We’ll define inputs, weights, and a bias, compute a loss, and let autodiff figure out the gradients for us. This mimics what happens in real machine learning.
```python
 import torch

# Inputs (2 samples, 3 features each)
X = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# True outputs (2 samples)
y_true = torch.tensor([1.0, 0.0])

# Parameters (weights and bias, with gradients enabled)
W = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)  # 1x3
b = torch.tensor([0.0], requires_grad=True)            # 1x1

# Forward pass: linear layer (y = XW^T + b)
y_pred = X @ W.T + b  # Matrix multiplication and broadcasting

# Mean squared error loss
loss = ((y_pred - y_true) ** 2).mean()

# Backward pass: compute gradients
loss.backward()

# Check the gradients
print(f"Loss: {loss.item()}")
print(f"Gradient of W: {W.grad}")
print(f"Gradient of b: {b.grad}")


# Output
# Loss: 4.045
# Gradient of W: tensor([[5.1000, 6.6000, 8.1000]])
# Gradient of b: tensor([2.1000])

```
- Here’s the breakdown:

- We’ve got two samples (X) and a linear layer with weights W and bias b.
- The forward pass computes predictions, then we calculate a loss (MSE).
- loss.backward() triggers autodiff, filling W.grad and b.grad with the gradients of the loss with respect to each parameter.

### Machine Learning with AutoDiff
- ![alt text](image-242.png)
- Machine Learning can be described as a 4 step process
- **Forward Pass**: We take our input variables(x) and we flow that input variable into our graph: x => m*x + b => y^(estimate of what y is)
- the estimate of y is supposed to be quite bad in the forward pass initially(since we selected random values of m and b)
- In the **second** step, we compare the y^(estimate of y) with the true value y to calculate cost C.
- Some people also refer to Cost as Loss Function also.
- ![alt text](image-243.png)
- Cost function compares y^ with y and gives a quantifiable value of cost value which is an index of how wrong the model is. 
- Remember our function has 3 parameters: x, m, b. Thay gives us y^
- Then we compare y^ with y and compute the cost value
- We have nested functions and we can use chain rule to calculate derivative of these nested functions.
- In Deep Learning Model we may have thousands of nested functions.
- In the **third** step, we use the chain rule to calculate the gradient of C with respect to the model parameters.
- ![alt text](image-244.png)
- In the **fourth step**, we adjust our parameters 'm' and 'b' to reduce our cost value C (or Loss Function)
- The way this works is because in step three, when we calculated the gradient or the slope of cost
with respect to our model parameters, this tells us, this slope tells us whether C is positively or
negatively related to M and B respectively.
- So let's say with M, if our cost with respect to M is positive, if the slope is positive, then we
know that if we were to reduce M somewhat then our cost would reduce as well.
- And same thing for B, so if cost with respect to B is say negatively related, if there's a negative
slope, then we know that if we increase B, then that will correspond to a reduction in C.
- ![alt text](image-245.png)
- Next we loop through these 4 steps till we reduce our cost function to close to zero. 
- This will help us get the real values of m and b and we can these use it to make predictions

```python
# The  y  values were created using the equation of a line  y=mx+b

x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.]) # E.g.: Dosage of drug for treating Alzheimer's disease
x #Output is tensor([0., 1., 2., 3., 4., 5., 6., 7.])


# The  y  values were created using the equation of a line  y=mx+b . This way, we 
# know what the model parameters to be learned are, say,  m=−0.5  and  b=2 . 
# Random, normally-distributed noise has been added to simulate sampling error:

# y = -0.5*x + 2 + torch.normal(mean=torch.zeros(8), std=0.2)

# For reproducibility of this demo, here's a fixed example of  y  values obtained # by running the commented-out line above:
y = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37]) # E.g.: Patient's "forgetfulness score"
y #Output is tensor([ 1.8600,  1.3100,  0.6200,  0.3300,  0.0900, -0.6700, -1.2300, -1.3700])

fig, ax = plt.subplots()
plt.title("Clinical Trial")
plt.xlabel("Drug dosage (mL)")
plt.ylabel("Forgetfulness")
_ = ax.scatter(x, y)

# Initialize the slope parameter  m  with a "random" value of 0.9...
# In this simple demo, we could guess approximately-correct parameter values to # start with. Or, we could use an algebraic (e.g., Moore-Penrose pseudoinverse) or # statistical (e.g., ordinary-least-squares regression) to solve for the parameters # quickly. This tiny machine learning demo with two parameters and eight data # points scales, however, to millions of parameters and millions of data points. # The other approaches -- guessing, algebra, statistics -- do not come close to # scaling in this way.)

m = torch.tensor([0.9]).requires_grad_()
m #Output is tensor([0.9000], requires_grad=True)

b = torch.tensor([0.1]).requires_grad_()
b #Output is tensor([0.1000], requires_grad=True)

def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

def regression_plot(my_x, my_y, my_m, my_b):
    
    fig, ax = plt.subplots()

    ax.scatter(my_x, my_y)
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, my_m, my_b).detach().item()
    y_max = regression(x_max, my_m, my_b).detach().item()
    
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])


regression_plot(x, y, m, b) 

# The above line is not so accurate, we will use autodiff to make it try to pass through these points

# Implement Machine Learning in 4 easy steps 

# Step 1 : Forward Pass
# For each of the 8 x-values, we get the y-values(note: they are not very accurate)
yhat = regression(x, m, b)
yhat #Output is tensor([0.1000, 1.0000, 1.9000, 2.8000, 3.7000, 4.6000, 5.5000, 6.4000],grad_fn=<AddBackward0>)

# Step 2 Compare yhat with true y to calculate cost C

# There is a PyTorch MSELoss method, but let's define it outselves to see how it works. MSE cost is defined by:
# Note that the bigger the difference between y^ and y, the more it gets amplified when we square it. So we are tolerant to small differences between them but the larger differences are highlighted more exponentially increase the cost. Thats why it is called Mean Squared error. Here n = 8
# C=(1/n)∑i=1n(y^i−yi)^2

def mse(my_yhat, my_y): 
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma/len(my_y)

C = mse(yhat, y)
C #Output is tensor(19.6755, grad_fn=<DivBackward0>)


#Step 3: Use autodiff to calculate gradient of  C  w.r.t. parameters
C.backward()
# If slope of C with respect to m or slope of C with respect to b is positive, reduce m and b as we need to move the slope to zero
m.grad #Output is tensor([36.3050]). Indicates slope/gradient is positive, we need to reduce m

b.grad #Output is tensor([6.2650]). Indicates slope is positive/gradient, we need to reduce b


# Step 4: Gradient descent
# Also called stochastic gradient descent
# We pass in the list of model parameters we like to adjust as first argument and we also pass the learning rate, which means how much we adjust model parameters ( m and b)
optimizer = torch.optim.SGD([m, b], lr=0.01)

# Adjust the values of m and b to reduce cost C, Remember m initially was 0.9 and b was 0.1
optimizer.step()

# Confirm parameters have been adjusted sensibly. Notice that values of m and b are adjusted to a lower value since the slope was positive.
m #Output is tensor([0.5369], requires_grad=True)

b #Output is tensor([0.0374], requires_grad=True)

regression_plot(x, y, m, b) #Output As we can see it has improved a little bit

# We can repeat steps 1 and 2 to confirm cost has decreased:
C = mse(regression(x, m, b), y)
C #Output is tensor(8.5722, grad_fn=<DivBackward0>)

# Put the 4 steps in a loop to iteratively minimize cost toward zero:
epochs = 1000
for epoch in range(epochs):
    
    optimizer.zero_grad() # Reset gradients to zero; else they accumulate
    
    yhat = regression(x, m, b) # Step 1
    C = mse(yhat, y) # Step 2
    
    C.backward() # Step 3
    optimizer.step() # Step 4
    
    print('Epoch {}, cost {}, m grad {}, b grad {}'.format(epoch, '%.3g' % C.item(), '%.3g' % m.grad.item(), '%.3g' % b.grad.item()))


```
- ![alt text](image-246.png)
- ![alt text](image-247.png)
- ![alt text](image-248.png)
- When we run the steps in a loop we get output like this 
- ![alt text](image-249.png)
- Notice that as cost decreases, the optimal values of 'm' and 'b' are being calculated.
- ![alt text](image-250.png)
- Finally we can plot the line as per optimal values of m and b like this 
```python
regression_plot(x, y, m, b)

m.item() #Output is -0.4681258499622345, remember correct value of m was -0.5
b.item() #Output is 1.7542961835861206, remember correct value of b was 2.0

# however we had added some sampling noise, but notice how close the estimates are. 
```
- ![alt text](image-252.png)
- The model doesn't perfectly approximate the slope (-0.5) and  y -intercept (2.0) used to simulate the outcomes  y  at the top of this notebook. This reflects the imperfectness of the sample of eight data points due to adding random noise during the simulation step. In the real world, the best solution would be to sample additional data points: The more data we sample, the more accurate our estimates of the true underlying parameters will be.

## Partial Derivative Calculus
- Use gradients in python to enable algorithms to learn from data.

### Review of Introductory Calculus 
- ![alt text](image-253.png)
- ![alt text](image-254.png)
- ![alt text](image-255.png)
- ![alt text](image-256.png)
- ![alt text](image-257.png)
- ![alt text](image-258.png)
- ![alt text](image-259.png)
- ![alt text](image-260.png)
- ![alt text](image-261.png)
- ![alt text](image-263.png)
- This section builds upon on the single-variable derivative calculus to introduce gradients and integral calculus. 
- Gradients of learning, which are facilitated by partial-derivative calculus, are the basis of training most machine learning algorithms with data -- i.e., stochastic gradient descent (SGD). 
- Paired with the principle of the chain rule (also covered in this class), SGD enables the backpropagation algorithm to train deep neural networks.
- Integral calculus, meanwhile, comes in handy for myriad tasks associated with machine learning, such as finding the area under the so-called “ROC curve” -- a prevailing metric for evaluating classification models. 

### Partial Derivatives
- ![alt text](image-264.png)

### Multivariate function 
- ![alt text](image-265.png)
- Therefore, we need to calculate partial derivative of m and partial derivative of b. 
- ![alt text](image-266.png)
- ![alt text](image-268.png)
- ![alt text](image-269.png)
- ![alt text](image-270.png)
- ![alt text](image-271.png)
- ![alt text](image-272.png)
- ![alt text](image-273.png)
- ![alt text](image-274.png)
- ![alt text](image-275.png)
- Determining partial derivatives by hand using rules is helpful for understanding how calculus works. In practice, however, autodiff enables us to do so more easily (especially if there are a large number of variables).
```python
# Remember z = x^2 - y^2
x = torch.tensor(2.).requires_grad_() 
x #Output is tensor(2., requires_grad=True)

y = torch.tensor(3.).requires_grad_() 
y #Output is tensor(3., requires_grad=True)

z = f(x, y) # Forward pass
z #Output is tensor(-5., grad_fn=<SubBackward0>)

z.backward() # Autodiff

x.grad #Output is tensor(4.)(Slope of z with respect to x)

y.grad #Output is tensor(-6.)(Slope of z with respect to y)

```
### Advanced Partial Derivatives
- ![alt text](image-276.png)
- We can prove this using PyTorch as follows:
```python
def cylinder_vol(my_r, my_l):
    return math.pi * my_r**2 * my_l

# Let's say the radius is 3 meters...
r = torch.tensor(3.).requires_grad_()
r #Output is tensor(3., requires_grad=True)

# ...and length is 5 meters:
l = torch.tensor(5.).requires_grad_()
l#Output is tensor(5., requires_grad=True)

# Then the volume of the cylinder is 141.4 cubic meters: 
v = cylinder_vol(r, l)
v #Output is tensor(141.3717, grad_fn=<MulBackward0>)

v.backward()

l.grad #Output is tensor(28.2743)

# Remember that change in length l by 1 unit results in change by pi*r^2

math.pi * 3**2 #Output is 28.274333882308138

# This means that with  value of r being constant at 3, a change in  l  by one unit corresponds to a change in  v  of 28.27 m3 . We can prove this to ourselves:

cylinder_vol(3, 6)

cylinder_vol(3, 6) - cylinder_vol(3, 5)
# Output is 28.274333882308127

cylinder_vol(3, 7) - cylinder_vol(3, 6)
# Output is 28.274333882308156

```

- Let us also calculate change with respect to radius also 
- ![alt text](image-277.png)
```python
# For changes in  v  with respect to  r, the volume changes by 2 *pi*r*l
r.grad #Output is tensor(94.2478)
2 * math.pi * 3 * 5 #Output is 94.24777960769379

delta = 1e-6

(cylinder_vol(3 + delta, 5) - cylinder_vol(3, 5)) / delta # Dividing by delta restores scale
#Output is 94.24779531741478
```

- $r$ is included in the partial derivative so adjusting it affects the scale of its impact on $v$. Although it's our first example in this notebook, it is typical in calculus for the derivative only to apply at an infinitesimally small $\Delta r$. The smaller the $\Delta r$, the closer to the true $\frac{\partial v}{\partial r}$. E.g., at $\Delta r = 1 \times 10^{-6}$:
- ![alt text](image-278.png)
- ![alt text](image-279.png)
- Notice in dz/dx, all other parameters are treated as constant so dz/dx(y^3) = 0;
- ![alt text](image-280.png)
- ![alt text](image-281.png)

### Partial Derivative Notation
- ![alt text](image-282.png)

### Applying Chain Rule to Partial Derivatives
- ![alt text](image-283.png)
- ![alt text](image-284.png)
- ![alt text](image-286.png)
- ![alt text](image-287.png)
- ![alt text](image-288.png)
- ![alt text](image-289.png)
- ![alt text](image-290.png)
- ![alt text](image-291.png)

### Point-by-Point Regression
- Recall the 4 step Machine Learning Loop
- ![alt text](image-292.png)
- ![alt text](image-293.png)
- we calculate the gradient of quadratic cost with respect to a straight-line regression model's parameters. We keep the partial derivatives as simple as possible by limiting the model to handling a single data point.
```python
import torch
xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

# The slope of a line is given by  y=mx+b :
def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

m = torch.tensor([0.9]).requires_grad_()
b = torch.tensor([0.1]).requires_grad_()

# To keep the partial derivatives as simple as possible, let's move forward with a single instance  i  from the eight possible data points:

i = 7
x = xs[i]
y = ys[i]

# Step 1: Forward pass
# We can flow the scalar tensor  x  through our regression model to produce  y^ , an estimate of  y . Prior to any model training, this is an arbitrary estimate:

yhat = regression(x, m, b)
yhat #Output tensor([6.4000], grad_fn=<AddBackward0>)

# Step 2: Compare  y^  with true  y  to calculate cost  C
# In the Regression in PyTorch notebook, we used mean-squared error, which averages quadratic cost over multiple data points. With a single data point, here we can use quadratic cost alone. It is defined by:C=(y^−y)^2
def squared_error(my_yhat, my_y):
    return (my_yhat - my_y)**2


C = squared_error(yhat, y)
C
# Output is tensor([60.3729], grad_fn=<PowBackward0>)

# Step 3: Use autodiff to calculate gradient of  C  w.r.t. parameters
C.backward()

# The partial derivative of  C  with respect to  m  ( ∂C/∂m ) is:
m.grad # Output is tensor([108.7800])

# And the partial derivative of  C  with respect to  b  ( ∂C/∂b ) is:
b.grad #Output is tensor([15.5400])

```
### Calculating Quadratic Cost w.r.t Predicted y
- ![alt text](image-294.png)
- ![alt text](image-295.png)
- ![alt text](image-296.png)
- ![alt text](image-297.png)

### The Gradient of Cost, $\nabla C$
- The gradient of cost, which is symbolized $\nabla C$ (pronounced "nabla C"), is a vector of all the partial derivatives of $C$ with respect to each of the individual model parameters: 
- $\nabla C = \nabla_p C = \left[ \frac{\partial{C}}{\partial{p_1}}, \frac{\partial{C}}{\partial{p_2}}, \cdots, \frac{\partial{C}}{\partial{p_n}} \right]^T $
- In this case, there are only two parameters, $b$ and $m$: 
- $\nabla C = \left[ \frac{\partial{C}}{\partial{b}}, \frac{\partial{C}}{\partial{m}} \right]^T $
```python
gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
gradient

#Output is tensor([[ 15.5400],
        #[108.7800]])
```
### Descending the Gradient of Cost
- Now we know what the gradient of cost is. Now we will descend the gradient
- ![alt text](image-298.png)
- ![alt text](image-299.png)
- Using partial derivatives, we now know how to calculate the gradient of cost with respect to parameters m and b.
- In Step 4, we need to descend the Gradient.(Ultimately the gradient of cost(loss function) must descend to 0). We need to adjust parameters(m and b) accordingly.
- ![alt text](image-300.png)

### Gradient of Cost on a batch of Data
- let's use mean squared error, which averages quadratic cost across multiple data points: $$C = \frac{1}{n} \sum_{i=1}^n (\hat{y_i}-y_i)^2 $$
```python
import torch
import matplotlib.pyplot as plt

xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

def regression(my_x, my_m, my_b):
    return my_m*my_x + my_b

m = torch.tensor([0.9]).requires_grad_()
b = torch.tensor([0.1]).requires_grad_()

# Step 1: Forward Pass
yhats = regression(xs, m, b)
yhats


def mse(my_yhat, my_y): 
    sigma = torch.sum((my_yhat - my_y)**2)
    return sigma/len(my_y)

# Step 2: Compare yhat with true y to calculate cost C
C = mse(yhats, ys)
C

# Step 3: Use autodiff to calculate gradient of  C  w.r.t. parameters

C.backward()
m.grad
b.grad


gradient = torch.tensor([[b.grad.item(), m.grad.item()]]).T
gradient

```
- ![alt text](image-301.png)
- ![alt text](image-302.png)
- $$ \frac{\partial C}{\partial m} = \frac{2}{n} \sum (\hat{y}_i - y_i) \cdot x_i $$
- $$ \frac{\partial C}{\partial b} = \frac{2}{n} \sum (\hat{y}_i - y_i) $$
```python

# Please note for below output is same as above for m.grad and b.grad
2*1/len(ys)*torch.sum((yhats - ys)*xs)

2*1/len(ys)*torch.sum(yhats - ys)

```
- ![alt text](image-303.png)
- $\frac{\partial C}{\partial m} = 36.3$ indicates that an increase in $m$ corresponds to a large increase in $C$. 

Meanwhile, $\frac{\partial C}{\partial b} = 6.26$ indicates that an increase in $b$ also corresponds to an increase in $C$, though much less so than $m$.

In the first round of training, the lowest hanging fruit with respect to reducing cost $C$ is therefore to decrease the slope of the regression line, $m$. There will also be a relatively small decrease in the $y$-intercept of the line, $b$. 

```python
optimizer = torch.optim.SGD([m, b], lr=0.01)
optimizer.step()
C = mse(regression(xs, m, b), ys)
```
- ![alt text](image-304.png)
- Do Further rounds of training 
```python
epochs = 8
for epoch in range(epochs): 
    
    optimizer.zero_grad() # Reset gradients to zero; else they accumulate
    
    yhats = regression(xs, m, b) # Step 1
    C = mse(yhats, ys) # Step 2
    
    C.backward() # Step 3
    
    labeled_regression_plot(xs, ys, m, b, C)
    
    optimizer.step() # Step 4


```
- In later rounds of training, after the model's slope $m$ has become closer to the slope represented by the data, $\frac{\partial C}{\partial b}$ becomes negative, indicating an inverse relationship between $b$ and $C$. Meanwhile, $\frac{\partial C}{\partial m}$ remains positive. 

This combination directs gradient descent to simultaneously adjust the $y$-intercept $b$ upwards and the slope $m$ downwards in order to reduce cost $C$ and, ultimately, fit the regression line snugly to the data. 
- ![alt text](image-305.png)
- With almost 1000 epochs of training, we get the following plot:
- ![alt text](image-307.png)

### Backpropagation
- ![alt text](image-308.png)

### Higher Order Partial Derivatives
- ![alt text](image-309.png)
- Higher Order Partial Derivatives are used in ML to accelerate through gradient descent.
- ![alt text](image-310.png)
- 2 types of second order partial derivatives: unmixed and mixed 
- ![alt text](image-311.png)
- ![alt text](image-312.png)
- ![alt text](image-313.png)
- Higher-order partial derivatives—those beyond the first derivative, like second (Hessians), third, or more—play a subtle but powerful role in machine learning. They’re not as front-and-center as first-order gradients (used in basic gradient descent), but they pop up in specific algorithms and scenarios where understanding curvature or higher-level behavior of the loss function gives you an edge.
- Optimization: Understanding Curvature with Second-Order Derivatives
- What They Are: The second partial derivatives form the Hessian matrix, which describes the local curvature of the loss function in multidimensional space.
- Use in ML: They help optimization algorithms go beyond the “slope” (first gradient) to understand how fast the slope changes—crucial for navigating tricky loss landscapes.
- Example: Newton’s Method and Quasi-Newton methods (like BFGS) use the Hessian (or approximations) to adjust step sizes and directions more intelligently than plain gradient descent. In a neural network, this can mean faster convergence by accounting for how parameters interact.
- Why It Matters: First-order methods (e.g., SGD) can zigzag or stall in flat regions or near saddle points. Second-order info helps “see” the terrain better, jumping over obstacles.
- Practical Catch: Computing the full Hessian is expensive—O(n²) storage and computation for n parameters—so it’s rare in deep learning with millions of parameters. Approximations (e.g., diagonal Hessian) or limited-memory methods (L-BFGS) are more common.
- In short, higher-order partial derivatives give ML a deeper view of the optimization problem—curvature, stability, uncertainty—at the cost of complexity. They’re like a high-powered lens: not always needed, but invaluable when precision matters.
- ![alt text](image-314.png)

## Integral Calculus
- ![alt text](image-315.png)
- ![alt text](image-316.png)
- ![alt text](image-317.png)
### Confusion Matrix
- A confusion matrix is a tool used in machine learning to evaluate the performance of a classification model by breaking down its predictions into a table. It shows how often the model’s predictions match the actual labels, giving you a clear picture of where it’s succeeding or screwing up. Think of it as a scorecard for your classifier—especially useful when you’re dealing with categories like “spam vs. not spam” or “cat vs. dog vs. bird.”
- It's called this because it's a matrix of when someone or an algorithm is confused as opposed to the
user of the matrix.
- It's more for logging.
- When some process leads to mistakes, including when an algorithm makes mistakes.
- For a binary classification problem (e.g., Positive vs. Negative), the confusion matrix is a 2x2 table:
- ![alt text](image-318.png)

### Receiver Operating Characteristic(ROC) curve
- It's an enormously useful metric for quantifying the performance of a binary classification model.
- Consider we have a binary classification algorithm to predict if there is a hotdog in an image or not. y corresponds to the actual outputs and y^ corresponds to the predictions made by the algorithm.
- ![alt text](image-319.png)
- Anything less than threshold is considered to be 0(not a hotdog) and above it is 1(it is a hotdog)
- ![alt text](image-322.png)
- ![alt text](image-323.png)
- The above is an ROC-AUC curve.
- Our objective is to have an algorithm that fills as much of the space under this curve as possible.
- So with our made up data points, our model currently has a area under the curve of 0.75, so we can
say an Roc, AUC, a receiver operating characteristic area under the curve of 0.75 now is 0.75.
- Is 0.75 good ?
- ![alt text](image-324.png)
- An ROC curve (Receiver Operating Characteristic curve) foundational questions about machine learningcurve) is a graphical tool used in machine learning to evaluate the performance of a binary classification model. It plots the trade-off between the model’s ability to correctly identify positive cases (True Positive Rate) versus its tendency to incorrectly label negative cases as positive (False Positive Rate) across different decision thresholds. It’s a go-to for understanding how well your classifier separates classes—like spam vs. not spam or disease vs. no disease—beyond a single accuracy score.

### Integral Calculus
- Study of areas under curves
- Facilitates the inverse of differential calculus.
- ![alt text](image-326.png)
- ![alt text](image-327.png)
- At a high level, how does integral calculus work?
- Much like differential calculus, it has to do with this idea of approaching infinity in some way.
- In the case of integral calculus, we use slices that correspond to rectangular area underneath a
curve.
- As the width of those slices approaches an infinitely small width that allows us to find the area
under the entirety of the curve.
- ![alt text](image-328.png)
- ![alt text](image-329.png)
- ![alt text](image-330.png)
- ![alt text](image-331.png)
- ![alt text](image-332.png)
- ![alt text](image-333.png)
- ![alt text](image-334.png)

### Definite Integrals
- Definite integral is not interested in entire area under curve but is interested in a specific area in the curve
- ![alt text](image-335.png)
- ![alt text](image-337.png)
- From the slides: $$ \int_1^2 \frac{x}{2} dx = \frac{3}{4} $$
```python
from scipy.integrate import quad # "quadrature" = numerical integration (as opposed to symbolic)
def g(x):
    return x/2
quad(g, 1, 2) #Output is (0.75, 8.326672684688674e-15)
# The second output of quad is an estimate of the absolute error of the integral, which in this case is essentially zero.

def h(x):
    return 2*x

quad(h, 3, 4) #Output is (7.0, 7.771561172376096e-14)

```
- ![alt text](image-338.png)


### Area undere ROC curve
- ![alt text](image-339.png)

```python
# When we don't have a function but we do have  (x,y)  coordinates, we can use the scikit-learn library's auc() method, which uses a numerical approach (the trapezoidal rule) to find the area under the curve described by the coordinates:

from sklearn.metrics import auc

# From the slides, the  (x,y)  coordinates of our hot dog-detecting ROC curve are:
xs = [0, 0,   0.5, 0.5, 1]
ys = [0, 0.5, 0.5, 1,   1]

auc(xs, ys) #Output is np.float64(0.75)


```
- ![alt text](image-340.png)

## Probability
- Quantifying Uncertainty and Building AI Systems that reason well despite it
- ![alt text](image-341.png)
- ![alt text](image-342.png)
- ![alt text](image-343.png)

### What is Probability Theory?
- Mathematical study of processes that include uncertainity.
- Expressed over a range of 0(will not happen) to 1 (will happen)
- ![alt text](image-344.png)

### Events and Sample Spaces

- Let's assume we have a fair coin, which is equally likely to come up heads (H) or tails (T).
- In instances like this, where the two outcomes are equally likely, we can use probability theory to express the likelihood of a particular **event** by comparing it with the **sample space** (the set of all possible outcomes; can be denoted as $\Omega$):
- $$ P(\text{event}) = \frac{\text{# of outcomes of event}}{\text{# of outcomes in }\Omega} $$
- If we're only flipping the coin once, then there are only two possible outcomes in the sample space $\Omega$: it will either be H or T (using set notation, we could write this as $\Omega = \{H, T\}$).
- Therefore: $$ P(H) = \frac{1}{2} = 0.5 $$
- Equally: $$ P(T) = \frac{1}{2} = 0.5 $$
- As a separate example, consider drawing a single card from a standard deck of 52 playing cards. In this case, the number of possible outcomes in the sample space $\Omega$ is 52.
- There is only one ace of spades in the deck, so the probability of drawing it is: $$ P(\text{ace of spades}) = \frac{1}{52} \approx 0.019 $$
- In contrast, there are four aces, so the probability of drawing an ace is: $$ P(\text{ace}) = \frac{4}{52} \approx 0.077 $$
- Some additional examples:
  - $$ P(\text{spade}) = \frac{13}{52} = 0.25 $$
  - $$ P(\text{ace OR spade}) = \frac{16}{52} \approx 0.307 $$
  - $$ P(\text{card}) = \frac{52}{52} = 1 $$
  - $$ P(\text{turnip}) = \frac{0}{52} = 0 $$

## Multiple Independent Observations

- Let's return to coin flipping to illustrate situations where we have an event consisting of multiple independent observations. For example, the probability of throwing two consecutive heads is: $$ P(\text{HH}) = \frac{1}{4} = 0.25 $$ ...because there is one HH event in the sample set of four possible events ($\Omega = \{HH, HT, TH, TT\}$).
- Likewise, the probability of throwing *three* consecutive heads is: $$ P(\text{HHH}) = \frac{1}{8} = 0.125 $$ ...because there is one HHH event in the sample set of eight possible events ($\Omega = \{HHH, HHT, HTH, THH, HTT, THT, TTH, TTT\}$).
- As final examples, the probability of throwing exactly two heads in three tosses is $$ P = \frac{3}{8} = 0.375 $$ while the probability of throwing at least two heads in three tosses is $$ P = \frac{4}{8} = 0.5 $$.
- In order to combine probabilities, we can multiply them. So the probability of throwing five consecutive heads, for example, is the product of probabilities we've already calculated: $$ P(\text{HHHHH}) = P(\text{HH}) \times P(\text{HHH}) = \frac{1}{4} \times \frac{1}{8} = \frac{1}{32} \approx 0.031 $$

## Combinatorics

- Combinatorics is a field of mathematics devoted to counting, and it can be helpful for studying probabilities.
- We can use **factorials** (e.g., $4! = 4 \times 3 \times 2 \times 1 = 24$), which feature prominently in combinatorics, to calculate probabilities instead of painstakingly determining all of the members of the sample space $\Omega$ and counting subsets within $\Omega$.
- More specifically, we can calculate the number of outcomes of an event using the "number of combinations" equation: $$ {n \choose k} = \frac{n!}{k!(n - k)!} $$
- The left-hand side of the equation is read "$n$ choose $k$" and is most quickly understood via an example: If we have three coin flips, $n = 3$, and if we're interested in the number of ways to get two head flips (or two tail flips, for that matter), $k = 2$. We would read this as "3 choose 2" and calculate it as:
  - $$ {n \choose k} = {3 \choose 2} = \frac{3!}{2!(3 - 2)!} = \frac{3!}{(2!)(1!)} = \frac{3 \times 2 \times 1}{(2 \times 1)(1)} = \frac{6}{(2)(1)} = \frac{6}{2} = 3 $$
- This provides us with the numerator for the event-probability equation from above: $$ P(\text{event}) = \frac{\text{# of outcomes of event}}{\text{# of outcomes in }\Omega} $$
- In the case of coin-flipping (or any binary process with equally probable outcomes), the denominator can be calculated with $2^n$ (where $n$ is again the number of coin flips), so: $$ \frac{\text{# of outcomes of event}}{\text{# of outcomes in }\Omega} = \frac{3}{2^n} = \frac{3}{2^3} = \frac{3}{8} = 0.375 $$

## Probability Exercises

### Exercises

1. What is the probability of drawing the ace of spades twice in a row? (Assume that any card drawn on the first draw will be put back in the deck before the second draw.)
2. You draw a card from a deck of cards. After replacing the drawn card back in the deck and shuffling thoroughly, what is the probability of drawing the same card again?
3. Use $n \choose k$ to calculate the probability of throwing three heads in five coin tosses.
4. Create a Python method that solves exercise 3 and incorporates the $n \choose k$ formula $\frac{n!}{k!(n - k)!}$. With the method in hand, calculate the probability of -- in five tosses -- throwing each of zero, one, two, three, four, and five heads.

### Solutions

1. $$ P(\text{ace of spades}) \times P(\text{ace of spades}) = \left(\frac{1}{52}\right)^2 = \frac{1}{2704} = 0.00037 = 0.037\% $$
2. $$ P(\text{any card}) = \frac{52}{52} = 1 $$
   $$ P(\text{same card as first draw}) = \frac{1}{52} \approx 0.019 $$
   $$ P(\text{any card})P(\text{same card as first draw}) = (1)\left(\frac{1}{52}\right) = \frac{1}{52} \approx 0.019 $$
3. $$ {n \choose k} = {5 \choose 3} = \frac{5!}{3!(5 - 3)!} = \frac{5!}{(3!)(2!)} = \frac{5 \times 4 \times 3 \times 2 \times 1}{(3 \times 2 \times 1)(2 \times 1)} = \frac{120}{(6)(2)} = \frac{120}{12} = 10 $$
   $$ P = \frac{10}{2^n} = \frac{10}{2^5} = \frac{10}{32} = 0.3125 $$
4. See the Python code below:

```python
from math import factorial

def coinflip_prob(n, k):
    n_choose_k = factorial(n) / (factorial(k) * factorial(n - k))
    return n_choose_k / 2**n

# Probability of 3 heads in 5 tosses
print(coinflip_prob(5, 3))  # Output: 0.3125

# Probabilities for 0 to 5 heads in 5 tosses
probabilities = [coinflip_prob(5, h) for h in range(6)]
print(probabilities)
# Output: [0.03125, 0.15625, 0.3125, 0.3125, 0.15625, 0.03125]
```
