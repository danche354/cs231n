# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:08:15 2016

@author: t-zopeng
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of per class
D = 2 # dimensionality
K = 3 # number of classes

X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels


for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral) # c: color, s: size

plt.show()

#%%

num_examples = X.shape[0]

# hypermeters
step_size = 1e-0
reg = 1e-3


# Train a Softmax Linear Classifier

# initialize parameters randomly
# size of hidden layer
h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))

W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

for i in range(20000):
    # evaluate the class scores with a 2-layer Neural Network
    # ReLU activation
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    
    scores = np.dot(hidden_layer, W2) + b2
    
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    
    # normalize them for each sample
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    correct_logprobs = -np.log(probs[range(num_examples), y])
    
    # compute the loss: average cross-entropy loss and regularizations
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W) + 0.5 * reg * np.sum(W2*W2)
    
    loss = data_loss + reg_loss
    
    if i%1000==0:
        print('iteration %d: loss %f'%(i, loss))
        # evaluate training set accuracy
        hidden_layer = np.maximum(0, np.dot(X, W) + b)
        scores = np.dot(hidden_layer, W2) + b2
        predicted_class = np.argmax(scores, axis=1)
        
        print('training accuracy: %.2f'%(np.mean(predicted_class==y)))
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    # backpropate thr gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer<=0] = 0
    
    # finalliy into W, b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    
    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W
    
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

#%%
# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())



















