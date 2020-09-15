import math
import random
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


weight_list = [random.random(), random.random(), random.random()]
# Where this is seen as: [w1, w2, bias]


def function(inp, w1=weight_list[0], w2=weight_list[1], bias=weight_list[2]):
    # Returns f(x) so we can keep our code slightly neater
    return w1*inp[0] + w2*inp[1] + bias


def state_correct(inp, desired_output):
    # I want this method to CHECK if y_i =? f(x_i)
    result = desired_output - function(inp)
    return result == 0


def weight_correction(inp, desired_output):
    # Where inp = a current slice of the array.
    # Correct our weights...
    # old_w1 = weight_list[0]
    # old_w2 = weight_list[1]
    weight_list[2] += desired_output - function(inp)
    for i in range(2):
        correction = (desired_output - function(inp))*inp[i]
        # correction = (desired_output - function(inp, w1 = old_w1, w2 = old_w2))*inp[i]
        weight_list[i] += correction
# Change: Commented out using old weights, because it's causing seriously bad overcorrection...


# Start our main below:
# And with our linear classifier
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.2, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s = 70)
# Messy, maybe clump all this graphing into a different method later?
xmin, xmax = 10, -10
for i in range(len(X)):
    if xmin > X[i][0]:
        xmin = X[i][0]
    if xmax < X[i][0]:
        xmax = X[i][0]
xmin = int(xmin)  # Since this just needs to be low, we can just round down with int()
xmax = math.ceil(xmax)  # This needs to rounded to the highest int, must use math.ceil()
x = list(range(xmax - xmin + 1))  # I found this to get long enough...

# Loop, until no change in weights seems like an easy way to enter an infinite loop...
# for i in range(len(X)):  # Where inp = input of [x_1, x_2] with known output y: [#]
#     while not state_correct(X[i], y[i]):
#         correct_or_no = state_correct(X[i], y[i])
#         if not state_correct(X[i], y[i]):
#             # We must do some corrections because f(x_i) != y
#             weight_correction(X[i], y[i])
#         else:
#             break
# Yep, it broke... iterative we go...
start = time.time()
iters = 0
while iters != 2000:
    for index in range(len(X)):
        if not state_correct(X[index], y[index]):
            f = function(X[index])
            weight_correction(X[index], y[index])
        else:
            break
    iters += 1
end = time.time()
print(f'{iters} iterations and {end-start} seconds have passed.')

# Graph line after all the corrections
slope = min(weight_list[0], weight_list[1]) / max(weight_list[0], weight_list[1])
    # TODO: Fix the slope formula (ignoring 0 values since super improbable...), gets messy as std increases
# NOTE: The line drawing isn't perfect, but it works fine about 80% of the time...
# Use the last corrected instance (after we already went through our iterations basically)
ys = []  # The y values we calculate for mapping the line part of the classifier
for inp in x:
    ys.append(slope*inp + (weight_list[2]/min(weight_list[0], weight_list[1])))
    # append: y = m*x_i + -bias/(what keeps it reasonable)

plt.plot(x, ys)  # Hopefully draws a line in the right place?
plt.show()
# Bottom Notes before I forget
# Ask how weights rebalance without overcorrecting?
# And what slope is?
# Definitely two biggest questions... for now I have to give up because I just don't see what to adjust... and why...
