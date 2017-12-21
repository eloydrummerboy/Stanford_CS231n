# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:15:10 2017

@author: scot274
"""

#Inputs:
#  - W: A numpy array of shape (D, C) containing weights.
#  - X: A numpy array of shape (N, D) containing a minibatch of data.
#  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
#    that X[i] has label c, where 0 <= c < C.
#  - reg: (float) regularization strength
    
          
         
import numpy as np
N = 50 # Training Examples
D = 3000 # Dimensions
C = 10 # Classes


W = np.random.randn(D,C)
X = np.random.randn(N,D)
y = np.random.randint(0,C,N)

reg_nv = 0.5
reg = reg_nv

# non-vectorized
dW_nv = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
num_classes_nv = W.shape[1]
num_train_nv = X.shape[0]
loss_nv = 0.0
count_missed_margin_nv = 0 # Count the number of classes that don't meet the margin
  # criteria. Used for Gradient.
  
for i in range(num_train_nv): # For each training example
  scores_nv = X[i].dot(W) # returns 1xC vector, ~prob that X[i] is of class c
  correct_class_score_nv = scores_nv[y[i]] # returns the ~prob for the correct class
  for j in range(num_classes_nv): # For each class
    if j == y[i]: # skip the correct class        
      continue
    margin_nv = scores_nv[j] - correct_class_score_nv + 1 # note delta = 1

    if margin_nv > 0:
      loss_nv += margin_nv 
      count_missed_margin_nv += 1
      dW_nv[:,j] += X[i]
      #dW[:,y[i]] -= X[i] <- Alternative to doing the single assignment below

  dW_nv[:,y[i]] -= count_missed_margin_nv*X[i]
  count_missed_margin_nv = 0


  
# Right now the loss is a sum over all training examples, but we want it
# to be an average instead so we divide by num_train.
loss_nv /= num_train_nv
  
# Add regularization to the loss.
loss_nv += reg_nv * np.sum(W * W)
  
dW_nv = np.divide(dW_nv,num_train_nv)
dW_nv = np.add(dW_nv,reg_nv*W)


############################
# Vectorized
loss = 0.0
dW = np.zeros(W.shape) # initialize the gradient as zero
num_train = X.shape[0]

scores = X.dot(W) # NxC (Training Examples x Classes)
correct_class_scores = scores[np.arange(0,num_train),y].reshape(-1,1)
margin = scores - correct_class_scores + 1
margin[np.arange(0,num_train), y] = 0 # Set correct score to 0
  
margin[margin<0] = 0       
loss = np.sum(margin)
# Take all margins that are greater than 1, sum them up
loss /= N # Dvide by number of training examples
loss += reg*np.sum(W*W) # add regularization



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
count_missed = np.sum(miss_margin, axis=1).reshape(-1,1) # find how many classes each sample missed
cm = np.zeros((N,C))
cm[np.arange(0,N),y] = -1*np.sum(miss_margin, axis=1)                    
dWyi = X.T.dot(cm)                     
# multiply all samples by this number
# now we need to subtract this from every column of dW that is the correct class for that sample
dW = dWyi + dWwj
dW /= num_train
dW += reg*W


# We want to go down each column of our margin matrix (samples x classes)
# if the particular sample did not meet the margin for that class, i.e.
# margin > 0, then add that sample (all D dimensions) to dW[class being examined]
# So in numpy terms, we want to sum all

# Let's first try with one Class (0)
miss_margin = margin > 0 # already accounts for j != yi
test = X[miss_margin[:,0]]
test = np.sum(X[miss_margin[:,0]],axis=0)
test /= num_train
test += reg*W[:,0]

  

test = miss_margin.T.dot(X)
test /= num_train
test += reg*W

count_missed = np.sum(miss_margin, axis=1).reshape(-1,1)
test = count_missed.T * X.T.dot(-1*miss_margin) + X.T.dot(miss_margin)

dWyi = np.sum(-count_missed*X,axis=0).reshape(-1,1)
dWwj = X.T.dot(miss_margin)

dWwj /= num_train
dWwj += reg*W

test /= num_train
test += reg*W
######



#L = scores - correct_scores + deltas
L = margin

L[L < 0] = 0
L[L > 0] = 1
L[np.arange(0, scores.shape[1]),y] = 0 # Don't count y_i
L[np.arange(0, scores.shape[1]),y] = -1 * np.sum(L, axis=0)
dW = np.dot(L, X.T)

  # Average over number of training examples
num_train = X.shape[1]
dW /= num_train


# Get batches
import numpy as np
N = 50 # Training Examples
D = 3000 # Dimensions
C = 10 # Classes


W = np.random.randn(D,C)
X = np.random.randn(N,D)
y = np.random.randint(0,C,N)

batch_size = 10

mask = np.random.choice(X.shape[0],batch_size,replace=False)
batch = X[mask]

y_pred = np.argmax(X.dot(W), axis=1)
y_predict = np.max(y_pred,axis = 1)


learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]

test1, test2 = np.meshgrid(learning_rates,regularization_strengths)



learning_rates = [3, 5, 7, 8]
regularization_strengths = [2, 4, 3]

test1, test2 = np.meshgrid(learning_rates,regularization_strengths)

tf1 = test1.flatten()
tf2 = test2.flatten()










