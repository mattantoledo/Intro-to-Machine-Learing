# Intro-to-Machine-Learning
Practical assignments of Introduction to Machine Learning course at TAU

### hw1 - k-Nearest Neighbours on MNIST dataset
- Fetch the MNIST dataset that contains 28x28 images of handwritten digits with their labels
- Implement the k-NN algorithm using different k and different sizes of the training set
- Compare the results to find the best k

### hw2 - Union of Intervals - ERM
- Create data set of 1D points in (0,1)
- Binary classification (+1, -1) based on k disjoint intervals
- Calculate empirical error and true error (distribution is given)
- Find the best k
- Holdout-Validation method

### hw3 - SGD (hinge loss vs. log loss) on MNIST dataset
- Implement SGD functions with appropriate gradient updates (hinge loss and log loss)
- Cross-validate to find best step size for SGD
- Calculate the accuracy of the classifiers

### hw4 - Kernel SVM
- Create data set of 2D points classified by a circle.
- Explore the SVM model with different kernels (linear, polynomial, RBF) and compare different parameters (coef0, gamma)
- Add some noise to the labels and see how the models behave

### hw5 - Training a Neural Network on the MNIST dataset using Back-Propagation 
- Implement the back-prop algorithm for the update of SGD with mini-batches
- Train one-hidden layer neural netwrk on the dataset
- Explore different learning rates for SGD

### hw6 - PCA
- Implement PCA without using pre-built functions, utilizing Numpy's SVD method to compute the top k eigenvectors and eigenvalues of the covariance matrix.
- Run PCA on a person's image matrix. 
- Plot the first 10 eigen-vectors as images, providing insights into common variations among the images.
- Use PCA to reduce image dimensions for different values of k. 
- Randomly select five images and compare the original and reconstructed versions. 
- Measure the sum of ℓ2 distances between them, indicating information loss in reconstruction.

## Photos

### hw1 - k-Nearest Neighbours on MNIST dataset
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw1/plots/knn_c.png "Graph 1c")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw1/plots/knn_d.png "Graph 1d")

### hw2 - Union of Intervals - ERM
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw2/plots/d.png "Graph 2d")

### hw3 - SGD (hinge loss vs. log loss) on MNIST dataset
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q1a.png "Graph 3.1a")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q1b.png "Graph 3.1b")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q1c.png "Graph 3.1c")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q2a.png "Graph 3.2a")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q2b.png "Graph 3.2b")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw3/plots/q2c.png "Graph 3.2c")

### hw4 - Kernel SVM
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw4/plots/q1.png "Graph 4.1")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw4/plots/q2.png "Graph 4.2")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw4/plots/q3.png "Graph 4.3")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw4/plots/q3a.png "Graph 4.3a")

### hw5 - Training a Neural Network on the MNIST dataset using Back-Propagation 
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw5/plots/b1.png "Graph 5.1")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw5/plots/b2.png "Graph 5.2")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw5/plots/b3.png "Graph 5.3")

### hw6 - PCA 
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw6/plots/b.png "Graph 6.1")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw6/plots/c10.png "Graph 6.2")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw6/plots/c50.png "Graph 6.3")
![](https://github.com/mattantoledo/Intro-to-Machine-Learning/blob/master/hw6/plots/c_k_values.png "Graph 6.3")
