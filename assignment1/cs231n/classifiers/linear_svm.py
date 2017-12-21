import numpy as np
from random import shuffle
#from past.builtins import xrange
from IPython import get_ipython

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  count_missed_margin = 0 # Count the number of classes that don't meet the margin
                          # criteria. Used for Gradient.
  
  for i in range(num_train): # For each training example
    scores = X[i].dot(W) # returns 1xC vector, ~prob that X[i] is of class c
    correct_class_score = scores[y[i]] # returns the ~prob for the correct class
    for j in range(num_classes): # For each class
      if j == y[i]: # skip the correct class        
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # Takes difference of the score for every other class and the correct score
      # Adds in a delta. The SVM "wants" the correct score to be greater than
      # all other scores by at least 'delta'. i.e. we want negative
      # margin. (see line below)
      if margin > 0:
        loss += margin # For all "bad" margins (not diff not > delta value)
                       # add them up, that's our loss, which we want to minimize
        count_missed_margin += 1
        # Gradient for all columns of the wrong class (if margin in not within
        # limit) is X[i]
        dW[:,j] += X[i]
        #dW[:,y[i]] -= X[i] <- Alternative to doing the single assignment below

    # Gradient for the column with the correct class is simply X[i]
    # scaled by the number of missed classes.
    dW[:,y[i]] -= count_missed_margin*X[i]
    count_missed_margin = 0

    
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW = np.divide(dW,num_train)
  dW = np.add(dW,reg*W)
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W) # NxC (Training Examples x Classes)
  correct_class_scores = scores[np.arange(0,num_train),y].reshape(-1,1)

  margin = scores - correct_class_scores + 1
  margin[np.arange(0,num_train), y] = 0 # Set correct score to 0
  margin[margin<0] = 0 # set margins that "pass" to zero, so they don't 
                       # affect the loss.
  loss = np.sum(margin)
  # Take all margins that are greater than 0, sum them up
  loss /=  num_train # Dvide by number of training examples
  loss += reg*np.sum(W*W) # add regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #Gradient
    # dW has C columns, one for each class
    # The gradient is as such:
    # For each Xi, we add it to the column if:
        # it doesn't pass the margin test for that row
        # AND
        # that row isn't the correct row for that sample
    # Above we've already created a matrix "margin" where
    # the dimensions are samples-by-classes, that indicates the margin
    # of each sample for each class, but is set to 0 where the margin
    # passed (i.e. was negative) or where that class was the correct class
    # so we only need to create a mask matrix where every instance of 'margin'
    # that is > 0 = 1 (or True, which is equivalent in Python). Then we can 
    # dot this mask with X, which will essentially do the above.
  miss_margin = margin > 0 # already accounts for j != yi
  dWwj = X.T.dot(miss_margin)
    # We also count how many classes for which that sample didn't 
    # pass the margin test, call this Z
    # for the column of dW that corresponds to the correct class for Xi
    # we add -Z*Xi
  cm = np.zeros(miss_margin.shape)
  cm[np.arange(0,num_train),y] = -1*np.sum(miss_margin, axis=1)                    
  dWyi = X.T.dot(cm)                     
    # multiply all samples by this number
    # now we need to subtract this from every column of dW that is the correct class for that sample
  dW = dWyi + dWwj
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
