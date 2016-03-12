
import numpy as np
from theano import *
import theano.tensor as T
import pickle, gzip
import matplotlib.pyplot as plt

def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if p == a:
            correct += 1
    return correct / total

def encode_labels(labels, max_index):
    """Encode the labels into binary vectors."""
    # Allocate the output labels, all zeros.
    encoded = np.zeros((labels.shape[0], max_index + 1))
    
    # Fill in the ones at the right indices.
    for i in range(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded

# Load the dataset
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

print( 'Shapes:')
print( '\tTraining:   ', train_set[0].shape, train_set[1].shape)
print( '\tValidation: ', valid_set[0].shape, valid_set[1].shape)
print( '\tTest:       ', test_set[0].shape, test_set[1].shape)

# Initialize shared weight variables
W1_shape = (50, 784)
b1_shape = 50
W2_shape = (10, 50)
b2_shape = 10

W1 = shared(np.random.random(W1_shape) - 0.5, name="W1")
b1 = shared(np.random.random(b1_shape) - 0.5, name="b1")
W2 = shared(np.random.random(W2_shape) - 0.5, name="W2")
b2 = shared(np.random.random(b2_shape) - 0.5, name="b2")

# Symbolic inputs
x = T.dmatrix("x") # N x 784
labels = T.dmatrix("labels") # N x 10

# Symbolic outputs
hidden = T.nnet.sigmoid(x.dot(W1.transpose()) + b1)
output = T.nnet.softmax(hidden.dot(W2.transpose()) + b2)
prediction = T.argmax(output, axis=1)
reg_lambda = 0.0001
regularization = reg_lambda * ((W1 * W1).sum() + (W2 * W2).sum() + (b1 * b1).sum() + (b2 * b2).sum())
cost = T.nnet.binary_crossentropy(output, labels).mean() + regularization

# Output functions
compute_prediction = function([x], prediction)

# Training functions
alpha = T.dscalar("alpha")
weights = [W1, W2, b1, b2]
updates = [(w, w - alpha * grad(cost, w)) for w in weights]
train_nn = function([x, labels, alpha],
                 cost,
                 updates=updates)

alpha_r = 10.0
labeled = encode_labels(train_set[1], 9)

costs = []
while True:
    costs.append(float(train_nn(train_set[0], labeled, alpha_r)))

    if len(costs) % 10 == 0:
        print('Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha_r)
    if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
        if alpha_r < 0.2:
            break
        else:
            alpha_r = alpha_r / 1.5

prediction = compute_prediction(test_set[0])
accuracy(prediction, test_set[1])

val_W1 = W1.get_value()
activations = [val_W1[i, :].reshape((28, 28)) for i in range(val_W1.shape[0])]

for i, w in enumerate(activations):
    plt.subplot(5, 10, i + 1)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.imshow(w)
plt.subplots_adjust(hspace=-0.85)
plt.gcf().set_size_inches(9, 9)

