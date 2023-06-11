import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimesnions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
	plt.title('title', size=12)
	plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if (target == target_label):
			image_vector = image.reshape((h*w, 1))
			selected_images.append(image_vector)
	return selected_images, h, w

def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
	  U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
	  		of the covariance matrix.
	  S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	X -= np.mean(X, axis=0)
	C = np.cov(X, rowvar=False)

	u, sigma, vt = np.linalg.svd(C, full_matrices=False)
	U = vt[0:k, :]
	S = sigma[0:k]
	return U, S


def b():
	selected_images, h, w = get_pictures_by_name()
	X = np.array(selected_images)[:,:,0]

	U, S = PCA(X, k=10)
	fig = plt.figure(figsize=(10, 6))

	for i in range(U.shape[0]):
		fig.add_subplot(2, 5, i+1)
		plt.imshow(U[i - 1].reshape((h, w)), cmap=plt.cm.gray)
		plt.title('vector:' + str(i), size=12)
	plt.show()


b()


def c():
	selected_images, h, w = get_pictures_by_name()
	X = np.array(selected_images)[:, :, 0]

	k_values = [1,5,10,30,50,100]

	sum_l2_distances = []

	for k in k_values:

		U, S = PCA(X, k)

		X_reduced = X @ U.T
		X_restored = X_reduced @ U

		X_restored += np.mean(X, axis=0)

		random_indices = np.random.choice(len(selected_images), size=5, replace=False)
		curr_sum_distances = 0

		fig = plt.figure(figsize=(4, 8))
		j = 0
		for i in random_indices:
			fig.add_subplot(5, 2, 2 * j + 1)
			plt.imshow(X[i].reshape((h, w)), cmap=plt.cm.gray)
			fig.add_subplot(5, 2, 2 * j + 2)
			plt.imshow(X_restored[i].reshape((h, w)), cmap=plt.cm.gray)

			j += 1
			l2_distance = np.linalg.norm(X[i] - X_restored[i])
			curr_sum_distances += l2_distance

		sum_l2_distances.append(curr_sum_distances)
		plt.suptitle('k: ' + str(k) + ',   original : transformed')
		plt.show()

	plt.plot(k_values, sum_l2_distances, marker='o')
	plt.xlabel('k')
	plt.ylabel('Sum of ℓ2 Distances')
	plt.title('Sum of ℓ2 Distances for Different k Values')
	plt.show()


c()
