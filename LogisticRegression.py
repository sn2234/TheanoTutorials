
import numpy as np
from theano import *
import theano.tensor as T
import pickle, gzip
import matplotlib.pyplot as plt

def encode_labels(labels, max_index):
    """Encode the labels into binary vectors."""
    # Allocate the output labels, all zeros.
    encoded = np.zeros((labels.shape[0], max_index + 1))
    
    # Fill in the ones at the right indices.
    for i in range(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if p == a:
            correct += 1
    return correct / total

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

print( 'Shapes:')
print( '\tTraining:   ', train_set[0].shape, train_set[1].shape)
print( '\tValidation: ', valid_set[0].shape, valid_set[1].shape)
print( '\tTest:       ', test_set[0].shape, test_set[1].shape)

# Weight vector shape: from 784 pixels to 10 possible classifications
W_shape = (10, 784)
b_shape = 10

W = shared(np.random.random(W_shape) - 0.5, name="W")
b = shared(np.random.random(b_shape) - 0.5, name="b")

x = T.dmatrix("x") # N x 784
labels = T.dmatrix("labels") # N x 10

output = T.nnet.softmax(x.dot(W.transpose()) + b)

prediction = T.argmax(output, axis=1)

reg_lambda = 0.001
cost = T.nnet.binary_crossentropy(output, labels).mean()
regularized_cost = cost + reg_lambda * ((W * W).sum() + (b * b).sum())

print('Example label encoding')
print(encode_labels(np.array([1, 3, 2, 0]), 3))

compute_prediction = function([x], prediction)
compute_cost = function([x, labels], cost)

# Compute the gradient of our error function
grad_W = grad(regularized_cost, W)
grad_b = grad(regularized_cost, b)

# Set up the updates we want to do
alpha = T.dscalar("alpha")
updates = [(W, W - alpha * grad_W),
           (b, b - alpha * grad_b)]

# Make our function. Have it return the cost!
train_regularized = function([x, labels, alpha],
                             regularized_cost,
                             updates=updates)

labeled = encode_labels(train_set[1], 9)

costs = []
alpha_r = 10
while True:
    costs.append(float(train_regularized(train_set[0], labeled, alpha_r)))
    
    if len(costs) % 10 == 0:
        print('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha_r)
    if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
        if alpha_r < 0.2:
            break
        else:
            alpha_r = alpha_r / 1.5

prediction = compute_prediction(test_set[0])

testAcc = accuracy(prediction, test_set[1])

print("Accuracy: {0}".format(testAcc))

val_W = W.get_value()
activations = [val_W[i, :].reshape((28, 28)) for i in range(val_W.shape[0])]


for i, w in enumerate(activations):
    plt.subplot(1, 10, i + 1)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(w)
plt.gcf().set_size_inches(15, 15)
