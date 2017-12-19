# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:00:45 2017

@author: scot274
"""
class Solution:
    

    def __init__(self):
        self.mem_cache = {}
        print("In __init__\n")
        
    def uniquePaths(self, m, n):
            # write your code here
            cache_idx = "%s,%s" %(m,n) 
            cache_idx_rev = "%s,%s" %(n,m) 
            if cache_idx in self.mem_cache:
                return self.mem_cache[cache_idx]
            elif m == 1 or n == 1:
                return 1
            else:
                val = self.uniquePaths(m-1,n) + self.uniquePaths(m, n-1)
                self.mem_cache[cache_idx] = val
                if cache_idx_rev not in self.mem_cache:              
                    self.mem_cache[cache_idx_rev] = val
                return val
s = Solution() 
print(s.uniquePaths(6,63))
print("\n")


num_training = 5000
mask = list(range(num_training))

import numpy as np
#labels (of group 1-5) for 24 training examples
y_train = [1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,3,4,4,3,5,5,1,1,1]

numtests = 3
dists = np.random.randn(numtests,len(y_train))**2

# Predict Labels
closest_y = []

closest_y = [y_train[x] for x in list(dists[1].argsort()[:3])]                     
   
max_count = 0
cur_count = 0
closest_y.sort(reverse=True) # Was instructed to keep smallest in case of ties
                             # so checking the larger indexes first.
most_label = closest_y[0]                   
for x in range(len(closest_y)):
    print(closest_y[x])
    print(closest_y.count(closest_y[x]))
    cur_count = closest_y.count(closest_y[x]) # Count number of time the value in the Xth position shows up
    if cur_count >= max_count: # If there are most of these than the current max
                               # check for equal, because we're supposed to keep smallest in case of tie
            most_label = closest_y[x]
            max_count = cur_count

X = np.array([[7,2,7,4,5],
              [7,3,7,5,6],
              [3,4,5,6,7]])
    
X_train = np.array([[1,2,3,4,5],
                    [2,3,4,5,6],
                    [4,4,5,6,7],
                    [7,7,7,7,7],
                    [8,4,8,4,8],
                    [10,10,11,2,7]])
    
## Two Loops  
dists_two = np.zeros((X.shape[0], X_train.shape[0]))
for i in range(X.shape[0]):
    for j in range(X_train.shape[0]):
        dists_two[i,j] = np.sqrt(np.sum(np.square(X[i] - X_train[j])))

## One Loop
dists_one = np.zeros((X.shape[0], X_train.shape[0]))
for i in range(X.shape[0]):
        dists_one[i,:] = np.sqrt(np.sum(np.square(X[i] - X_train),axis=1))


## No Loops
# Start with one row of X, then extrapolate to all of X below
Xsq = np.sum(X[2]**2,keepdims=True)
Xtsq = np.sum(X_train**2,axis=1,keepdims=True)
XsqplusXtsq = Xsq + Xtsq
#two_xt = 2*np.matmul(X[0],X_train.T).reshape(3,1)
#two_xt = 2*X[2]*X_train
two_xt = 2*np.dot(X[2],X_train.T)
#d = Xsq + Xtsq - two_xt
d = Xsq + Xtsq - two_xt.T
d = np.sqrt(np.sum(d, axis=1))


# No Loops
Xsq = np.sum(X**2, axis=1,keepdims=True)
Xtsq = np.sum(X_train**2, axis=1,keepdims=True)
#two_xt = 2*np.matmul(X[0],X_train.T).reshape(3,1)
two_xt = 2*np.dot(X,X_train.T)
#d = Xsq + Xtsq - two_xt
d = Xsq.T + Xtsq - two_xt.T
d = np.sqrt(np.abs(d))


dists = np.sqrt(np.sum(np.square(X),axis=1, keepdims=True).T + np.sum(np.square(X_train),axis=1,keepdims=True) - 2*np.dot(X,X_train.T).T).T


min(X)

np.sum(np.sqrt(np.square(X[0] - X_train)),axis=1).T


listA = [0]
listB = listA
listB.append(1)
print(listA)
      
a = [4,5,1,2,3]

def shiftleft(array):
    temp=array[0]
    for x in range(len(array)-1):
        array[x] = array[x+1]
    array[len(array)-1] = temp
            
def shiftright(array):
    temp=array[len(array)-1]
    for x in range(len(a)-1,0-1,-1):
        array[x] = array[x-1]
    array[0] = temp
  
print(a)    
shiftleft(a)
print(a)
shiftleft(a)
print(a)  
shiftright(a)
print(a)

# Folds
X_train = np.array([[1,2,3,4,5],
                    [2,3,4,5,6],
                    [4,4,5,6,7],
                    [7,7,7,7,7],
                    [8,4,8,4,8],
                    [10,10,11,2,7]])

XTF = np.array(np.array_split(X_train,3))
Xtr = XTF[np.arange(len(XTF))!=1].reshape(-1,X_train.shape[1])
Xval = XTF[1]



