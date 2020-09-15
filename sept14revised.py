# simple file to generate groups of points with particular distributions
# for examining what happens in a simple learning paradigm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numpy.random import randint as rand
from time import time_ns as time
import math
from sklearn import datasets


# grabs
# [INFO] used only in plotDataSet()
def getValues(X, y, selectedLabel=0) :
	x1s = []
	x2s = []
	zeroLabels  = []
	for sample, label in zip(X, y) :
		[x1, x2] = sample
		if label == selectedLabel :
			x1s.append(x1)
			x2s.append(x2)
			zeroLabels.append(label)
	return x1s, x2s, zeroLabels

# Plots the dataset array of samples 'X' and corresponding label array 'y'
def plotDataSet(X, y) :
	x1s, x2s, labels = getValues(X, y, selectedLabel = 1)
	plt.plot(x1s, x2s, "go")
	x1s, x2s, labels = getValues(X, y, selectedLabel = 0)
	plt.plot(x1s, x2s, "ro")

# The mathematical function sign (used in predict())
def sign(x) :
	return 1 if x > 0 else 0

# Trains the dataset array of samples 'X' and corresponding label array 'y' off of specified weights
#	[Psuedocode]
# 	Set initial bias,weights (set to zero, random, ...)
# 	For some amount of time, or number of iterations, or until no change in weights
# 		For each input training example xn (with known label yn)
# 			If the yn minus your calculation of what it should (f(xn)) be is zero (you are correct), continue
# 			Else update weight wn associated with xn by adding to it (yn – f(xn))*xn
def train(weights, X, y, iterations=10) :
	start = time()  # capture the start time in nanoseconds
	for i in range(iterations) :   # perform training for a given number of iterations
		for sample, label in zip(X, y) :   # for each sample and its corresponding label do the following:
			diff = label - predict(weights, sample)    # check if their is a difference of the label and the prediction
			if diff == 0 :
				continue
			else :
				[x1, x2] = sample # fetch the sample's components
				[w1, w2, b] = weights  # fetch the weight's components

				weights[0] = w1 + diff * x1  # update the first weight
				weights[1] = w2 + diff * x2  # update the second weight
				weights[2] = b + diff # I consider b a weight
	end = time() # capture the end time in nanoseconds
	return end - start # return the difference in time


# Predicts a sample based on the linear separator defined by 'weights'
def predict (weights, sample) :
	[x1, x2] = sample # fetch the sample's components
	[w1, w2, b] = weights  # fetch the weight's components
	return sign( x1*w1 + x2*w2 + b ) # return the prediction

# Tests the dataset array of samples 'X' predictions against the true label array 'y' off of specified weights
def test(weights, X, y) :
	totalSize = len(X)
	correct = 0
	for x, ya in zip(X, y) :
		if predict(weights, x) == ya :		
			correct += 1
	return correct / totalSize


# Finding min and max of data sets to set line length correctly
def linelength(X) :
	minX = 0
	maxX = 0
	for i in range(len(X)):
		if X[i][0] < minX:
			minX = X[i][0]
		elif X[i][0] > maxX:
			maxX = X[i][0]
		else:
			minX = minX
			maxX = maxX
	minX = math.floor(minX)
	maxX = math.ceil(maxX)
	return minX,maxX


# Plots a separator based on the weights given
def plotSeparator(weights, start=-10, end=10) :
	checks = [ 
				(weights == None, "[ERROR] Weights are not initialized."),
			 	(len(weights) != 3, "[ERROR] This must be a 2D classifier.")
			 ]
	for check, msg in checks :
		if check :
			raise Exception(msg)
	[w1, w2, b] = weights
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






# Main function
if __name__ == "__main__" :

	# PART 1 ------------------------------------------------------
	response = input('Provide a number to use as a seed: ') # Set Seed
	SEED = rand(response) 
	STD  = .2  # Set Standard Deviation
	ITERATIONS = 2  # Set Number of iterations

	# Import the iris dataset to play with
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	y = iris.target
	print(y)

	# Initialize Weights :
	w1 = 0
	w2 = 0
	b = 0
	weights = [w1, w2, b] 

	# Train on generated data
	elapsedTime = train(weights, X, y, iterations = ITERATIONS)
	print(f"Time elapsed: {elapsedTime / 1000000} ms")
	
	# Plot Dataset :
	plotDataSet(X, y)

	# Plot Linear Separator :
	minX, maxX = linelength(X)
	plotSeparator(weights, start=minX, end=maxX)
	
	# Show Plot
	plt.show()

	
	