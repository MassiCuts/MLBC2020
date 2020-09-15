# simple file to generate groups of points with particular distributions
# for examining what happens in a simple learning paradigm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numpy.random import randint as rand
from time import time_ns as time

# grabs
# [INFO] used only in plotDataSet()
def getValues(X, y, selectedLabel=0) :
	zeroSamples = []
	zeroLabels  = []
	for sample, label in zip(X, y) :
		if label == selectedLabel :
			zeroSamples.append(sample)
			zeroLabels.append(label)
	return zeroSamples, zeroLabels

# plots
def plotDataSet(X, y) :
	samples, labels = getValues(X, y, selectedLabel = 1)
	plt.plot(samples, labels, "go")
	samples, labels = getValues(X, y, selectedLabel = 0)
	plt.plot(samples, labels, "ro")

# the mathematical function sign :
def sign(x) :
	return 1 if x > 0 else 0

def train(X, y, iterations=10, weights) :
	start = time()  # capture the start time in nanoseconds
	for i in range(iterations) :   # perform training for a given number of iterations
		for sample, label in zip(X, y) :   # for each sample and its corresponding label do the following:
			diff = label - predict(sample, weights)    # check if their is a difference of the label and the prediction
			if diff == 0 :
				continue
			else :
				[x1, x2] = sample # fetch sample components
				[w1, x2, b] = weights  # fetch weight components

				weights[0] = w1 + diff * x1  # update the first weight
				weights[1] = w2 + diff * x2  # update the second weight
				weights[2] = b + diff # I consider b a weight
	end = time() # capture the end time in nanoseconds
	return end - start # return the difference in time


def predict (sample, weights) :
	


# Set initial bias,weights (set to zero, random, ...)
# For some amount of time, or number of iterations, or until no change in weights
# 	For each input training example xn (with known label yn)
# 		If the yn minus your calculation of what it should (f(xn)) be is zero (you are correct), continue
# 		Else update weight wn associated with xn by adding to it (yn – f(xn))*xn
class LinearClassifier :
	def __init__(self) :
		self.weights = None

	def train(self, X, y, epochs = 10, initialWeights = None) :
		start = time()
		if initialWeights != None :
			self.weights = initialWeights
		elif self.weights == None:
			size = len(X[0]) + 1
			self.weights = [0] * size
		
		for _ in range(epochs) :
			for x, ya in zip(X, y) :
				diff = ya - self.predict(x)
				if diff == 0 :
					continue
				else :
					x = list(x)
					x.append(1)
					self.weights = [ w + diff * x for w, x in zip(self.weights, x)]
		end = time()
		return end - start

	def plot2DSeparator(self, start=-10, end=10) :
		checks = [ 
					(self.weights == None, "[ERROR] Weights are not initialized."),
				 	(len(self.weights) != 3, "[ERROR] This must be a 2D classifier.")
				 ]
		for check, msg in checks :
			if check :
				raise Exception(msg)
 
		[w1, w2, b] = self.weights
		if w1 == 0 and w2 == 0:
			print("[WARNING] Cannot plot separator: Both weights are zero.")
			return

		if w2 == 0 :
			m = - w2 / w1
			b = - b / w1
			y = list(range(start, end))
			x = [m * yi + b for yi in y]
			placement = "to the left"
		else :
			m = - w1 / w2
			b = - b / w2
			x = list(range(start, end))
			y = [m * xi + b for xi in x]
			if w2 < 0 :
				placement = "below"
			else :
				placement = "above"

		print(f"[INFO] Green is {placement} the line")

		plt.plot(x, y, 'b-')
	
	def test(self, X, y) :
		totalSize = len(X)
		correct = 0
		for x, ya in zip(X, y) :
			if self.predict(x) == ya :		
				correct += 1
		return correct / totalSize

	def predict(self, x) :
		x = list(x)
		x.append(1)
		result = sum([x*w for x, w in zip(x, self.weights)])
		return sign(result)

if __name__ == "__main__" :
	SEED = rand(1000)
	STD  = .2
	ITERATIONS = 1

	# PART 1
	X, y = make_blobs(n_samples=100, centers=2, cluster_std=STD, random_state=SEED)
	plotDataSet(X, y)
	
	classifier = LinearClassifier()
	elapsedTime = classifier.train(X, y, epochs = ITERATIONS)
	print(f"Time elapsed: {elapsedTime / 1000000} ms")
	classifier.plot2DSeparator(start = -3, end = 3)
	plt.show()


	# PART 2
	stds = [ *([0.2] * 4), *([0.4] * 4), *([0.6] * 4), *([0.8] * 4) ]
	THRESHOLD_ITERATIONS = 2000


	for run, std in zip(range(1, len(stds) + 1), stds) :
		seed = rand(1000)
		X, y = make_blobs(n_samples=100, centers=2, cluster_std=std, random_state=seed)
		classifier = LinearClassifier()
		elapsedTime = 0
		for i in range(1, THRESHOLD_ITERATIONS + 1) :
			elapsedTime += classifier.train(X, y, epochs = 1)
			if classifier.test(X, y) == 1 :
				break
		print(f"Run: {run} | STD: {std} | Iterations: {i} | Elapsed Time: {elapsedTime} ns")

