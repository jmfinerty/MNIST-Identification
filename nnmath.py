import numpy as np


# Return a np array of weights between 0 and 1
#  for a layer with from_size nodes
#  to a layer of to_size nodes
# Rows are w01, w11, w12... Cols are w01, w02, w03...
#  Result has to_size rows, from_size cols
#   This way, with x as a col vector, y = wx instead of wTx
def rand_weights(from_size, to_size):
    w = np.array(np.random.uniform(-1.0, 1.0, from_size))
    for _ in range(to_size - 1):  # first row already done
        w = np.vstack([w, np.random.uniform(-1.0, 1.0, from_size)])
    return w


# adds bias terms to read-in data
def add_bias(values, bias=1):
    new_values = []
    for row in values:
        new_values.append([bias] + row)
    return new_values


# sigmoid function
def sig(x):
    e = np.exp(1)
    return float(1 / (1 + e**(-1*x)))


# sigma'(x): first derivative of sigmoid function
def sigp(x):
    s = sig(x)
    return s * (1 - s)


# Returns activations of hidden nodes and output node
# based on data, current input layer weights, current hidden layer weights
def forward_pass(x, w_1, w_2):
    # hidden layer activations
    a_h = np.array([sig(i) for i in np.matmul(w_1, x)])
    a_o = sig(np.matmul(w_2, a_h))  # output layer activation -- scalar!!
    return a_h, a_o


# Backpropogation algorithm
# Given current state of NN, returns updated weights
def backprop(label, values, w_1, w_2, a_h, a_o, alpha):
    delta_o = (label - a_o) * sigp(a_o)
    delta_h = [sigp(a) for a in a_h] * w_2 * delta_o

    # update weights from h->o using delta_o
    w_2 = w_2 + alpha * delta_o * a_h
    # update weights from x->h using delta_h
    for row in range(len(w_1)):
        for col in range(row):
            w_1[row][col] = w_1[row][col] + alpha*delta_h[row]*values[col]

    return w_1, w_2
