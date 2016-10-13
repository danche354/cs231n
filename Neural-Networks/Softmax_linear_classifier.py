# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:06:39 2016

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
# Train a Softmax Linear Classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K) # 2*3
b = np.zeros((1, K))

num_examples = X.shape[0]

# hypermeters
step_size = 1e-0
reg = 1e-3

#%%
for i in range(200):
    # compute class scores for a linear classfier
    scores = np.dot(X, W) + b
    
    # get unnormalized probabilities
    exp_scores = np.exp(scores)
    
    # normalize them for each sample
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    correct_logprobs = -np.log(probs[range(num_examples), y])
    
    # compute the loss: average cross-entropy loss and regularizations
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W)
    
    loss = data_loss + reg_loss
    
    if i%10==0:
        print('iteration %d: loss %f'%(i, loss))
        # evaluate training set accuracy
        scores = np.dot(X, W) + b
        predicted_class = np.argmax(scores, axis=1)
        
        print('training accuracy: %.2f'%(np.mean(predicted_class==y)))
    
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    # backpropate the gradient to the parameters (W, b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    
    # regularition gradient
    dW += reg*W
    
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db

#%%
# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

