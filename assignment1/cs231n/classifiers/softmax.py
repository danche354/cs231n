import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    # exp scores
    exp_scores = np.exp(scores)
    # exp prob
    exp_prob = exp_scores / np.sum(exp_scores)
    
    #correct label
    correct_logprob = -np.log(exp_prob[y[i]])
    
    loss += correct_logprob
    
    dscores = exp_prob
    dscores[y[i]] -= 1
    dW += np.dot(X[i][:, None], dscores[None, :])
  
  # normalization
  loss /= num_train
  dW /= num_train
  
  # regularization
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  
  num_train = X.shape[0]

  scores = np.dot(X, W)
  exp_scores = np.exp(scores)
  exp_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  
  correct_problog= np.log(exp_prob[range(num_train), y])
  
  loss = -np.sum(correct_problog)  
  
  dscores = exp_prob
  dscores[range(num_train), y] -= 1

  dW = np.dot(X.T, dscores)
  
  # normalization
  loss /= num_train
  dW /= num_train
  
  # regularization
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

