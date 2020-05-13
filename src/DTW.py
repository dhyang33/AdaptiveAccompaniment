#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')
import cython
import numpy as np


# In[ ]:
def computeCostMatrix_cosdist(Fquery,Fref):
    '''
    Computes the cosine distance cost matrix.
    
    Arguments:
    Fquery -- the query chroma feature matrix of dimension (12, # query frames), this feature
              matrix is assumed to be L2 normalized
    Fref -- the reference feature matrix, of dimension (12, # reference frames), this feature
            matrix is assumed to be L2 normalized
    
    Returns:
    C -- cost matrix whose (i,j)th element specifies the cosine distance between the i-th query frame
         and the j-th reference frame
    '''
    
    ### START CODE BLOCK ###
    C = 1-Fquery.T@Fref
    ### START CODE BLOCK ###

    return C


# In[ ]:


def globalDTW(C, steps, weights, segment_lengths):
    '''
    Find the optimal subsequence path through cost matrix C.
    
    Arguments:
    C -- cost matrix of dimension (# query frames, # reference frames)
    steps -- a numpy matrix specifying the allowable transitions.  It should be of
            dimension (L, 2), where each row specifies (row step, col step)
    weights -- a vector of size L specifying the multiplicative weights associated 
                with each of the allowable transitions
                
    Returns:
    optcost -- the optimal subsequence path score
    path -- a matrix with 2 columns specifying the optimal subsequence path.  Each row 
            specifies the (row, col) coordinate.
    '''
    D = np.full(C.shape, np.inf)
    B = np.zeros((C.shape[0],C.shape[1],2))

    ### START CODE BLOCK ###
    inf = float("inf")
    minVal=inf
    for k in range(C.shape[1]):
        if C[0,k] < minVal:
            minVal = C[0,k]
        D[0,k]=minVal
    for i in range(1,D.shape[0]):
        for j in range(0,D.shape[1]):
            steps.append([1,int(segment_lengths[i]/2)])
            weights.append(1)
            opt=[inf for i in range(len(steps))]
            for index, s in enumerate(steps):
                if i-s[0] >= 0 and j-s[1]>=0 and D[i-s[0],j-s[1]]!=np.inf:
                    previousCell = D[i-s[0],j-s[1]]
                    opt[index] = previousCell +C[i,j]*weights[index]
            optimal = min(opt)
            opt_index = np.argmin(np.array(opt))
            D[i][j]=optimal
            B[i][j][0]=steps[opt_index][0]
            B[i][j][1]=steps[opt_index][1]
            if len(weights)>1:
              steps = [[0,1]]
              weights = [0]
    optcost = min(D[-1])
    path = global_backtrace(D,B,steps)
    
    ### END CODE BLOCK ###
    
    #path = np.array(path)
    return D, path


# In[ ]:


def global_backtrace(D, B, steps):
    '''
    Backtraces through the cumulative cost matrix D.
    
    Arguments:
    D -- cumulative cost matrix
    B -- backtrace matrix
    steps -- a numpy matrix specifying the allowable transitions.  It should be of
            dimension (L, 2), where each row specifies (row step, col step)
    
    Returns:
    path -- a python list of (row, col) coordinates for the optimal path.
    '''

    path = []

    ### START CODE BLOCK ###
    c = np.argmin(D[-1])
    r = B.shape[0]-1
    path = []
    while r!=0:
        path.append((r,c))
        step = B[r,c]
        r = int(np.round(r - step[0]))
        c = int(np.round(c - step[1]))
        
    path.append((r,c))
    
    ### END CODE BLOCK ###
    
    return path


# In[ ]:


def local_backtrace(D, C, endpoint):
    '''
    Backtraces through the cumulative cost matrix D.
    
    Arguments:
    D -- cumulative cost matrix
    B -- backtrace matrix
    steps -- a numpy matrix specifying the allowable transitions.  It should be of
            dimension (L, 2), where each row specifies (row step, col step)
    
    Returns:
    path -- a python list of (row, col) coordinates for the optimal path.
    '''

    path = []
    steps = np.array([2, 1, 1, 2, 1, 1]).reshape((-1,2))
    weights = [2,1,1]

    ### START CODE BLOCK ###
    r = C.shape[0]-1
    c = endpoint
    path = []
    while r!=0:
        path.append((r,c))
        for idx,step in enumerate(steps):
          prev_r = r - step[0] 
          prev_c = c - step[1]
          cost = C[r,c]
          accum_cost = D[r,c]
          prev_cell = D[prev_r,prev_c]
          if np.abs(prev_cell+cost*weights[idx]-accum_cost)<0.000001:
            r = prev_r
            c = prev_c
            break 
        
        if r==0:
          path.append((r,c))
    ### END CODE BLOCK ###
    
    return path


# In[ ]:


'''
def extrapolateTimeStretchFunction(local_paths, audio):
  new_stretch = []
  k = 0
  slope = 1
  new_time = 0
  counter = 0
  for i in range(0,len(audio)):
    if i >= local_paths[k][0] && i <= local_paths[k][1]:
      path = local_paths[k]

      slope = (path[c+1][1]-stretch_function[c][1])/(stretch_function[c+1][0]-stretch_function[c][0])
      k+=1
    else:
      slope=1
    new_time+=slope
    new_stretch.append((i, int(new_time)))
  return result
'''


# In[ ]:




