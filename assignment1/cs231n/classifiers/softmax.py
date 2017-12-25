import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_training = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_training):
    f = X[i].dot(W)
    f -= np.max(f)
    expsum = np.sum(np.exp(f))
    loss += -f[y[i]] + np.log(expsum)
    dW[:, y[i]] -= X[i]
    for j in range(num_classes):
      dW[:, j] += 1 / expsum * X[i] * np.exp(f[j])

  loss /= num_training
  loss += reg * np.sum(W * W)
  dW /= num_training
  dW += 2 * reg * W
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
  num_training = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  scores = scores - np.max(scores, axis=1)[:,np.newaxis] # stabilize
  exp = np.exp(scores)
  expsum = np.sum(exp, axis=1)
  loss = np.mean(-scores[np.arange(num_training), y] + np.log(expsum))
  loss += reg * np.sum(W * W)
  mask = np.zeros_like(exp)
  mask[np.arange(num_training), y] = 1
  vec = exp / expsum[:, np.newaxis] - mask 
  dW = X.T.dot(vec)

  dW /= num_training
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

