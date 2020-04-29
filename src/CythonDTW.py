#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'Cython')
import cython
import numpy as np


# In[ ]:


get_ipython().run_cell_magic('cython', '', 'import numpy as np\ncimport numpy as np\ncimport cython\n\nimport sys\nimport time\n\n\nDTYPE_INT32 = np.int32\nctypedef np.int32_t DTYPE_INT32_t\n\nDTYPE_FLOAT = np.float64\nctypedef np.float64_t DTYPE_FLOAT_t\n\ncdef DTYPE_FLOAT_t MAX_FLOAT = float(\'inf\')\n\n# careful, without bounds checking can mess up memory - also can\'t use negative indices I think (like x[-1])\n@cython.boundscheck(False) # turn off bounds-checking for entire function\ndef DTW_Cost_To_AccumCostAndSteps(Cin, parameter):\n    \'\'\'\n    Inputs\n        C: The cost Matrix\n    \'\'\'\n\n\n    \'\'\'\n    Section for checking and catching errors in the inputs\n    \'\'\'\n\n    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] C\n    try:\n        C = np.array(Cin, dtype=DTYPE_FLOAT)\n    except TypeError:\n        print(bcolors.FAIL + "FAILURE: The type of the cost matrix is wrong - please pass in a 2-d numpy array" + bcolors.ENDC)\n        return [-1, -1, -1]\n    except ValueError:\n        print(bcolors.FAIL + "FAILURE: The type of the elements in the cost matrix is wrong - please have each element be a float (perhaps you passed in a matrix of ints?)" + bcolors.ENDC)\n        return [-1, -1, -1]\n\n    cdef np.ndarray[np.uint32_t, ndim=1] dn\n    cdef np.ndarray[np.uint32_t, ndim=1] dm\n    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dw\n    # make sure dn, dm, and dw are setup\n    # dn loading and exception handling\n    if (\'dn\'  in parameter.keys()):\n        try:\n\n            dn = np.array(parameter[\'dn\'], dtype=np.uint32)\n        except TypeError:\n            print(bcolors.FAIL + "FAILURE: The type of dn (row steps) is wrong - please pass in a 1-d numpy array that holds uint32s" + bcolors.ENDC)\n            return [-1, -1, -1]\n        except ValueError:\n            print(bcolors.FAIL + "The type of the elements in dn (row steps) is wrong - please have each element be a uint32 (perhaps you passed a long?). You can specify this when making a numpy array like: np.array([1,2,3],dtype=np.uint32)" + bcolors.ENDC)\n            return [-1, -1, -1]\n    else:\n        dn = np.array([1, 1, 0], dtype=np.uint32)\n    # dm loading and exception handling\n    if \'dm\'  in parameter.keys():\n        try:\n            dm = np.array(parameter[\'dm\'], dtype=np.uint32)\n        except TypeError:\n            print(bcolors.FAIL + "FAILURE: The type of dm (col steps) is wrong - please pass in a 1-d numpy array that holds uint32s" + bcolors.ENDC)\n            return [-1, -1, -1]\n        except ValueError:\n            print(bcolors.FAIL + "FAILURE: The type of the elements in dm (col steps) is wrong - please have each element be a uint32 (perhaps you passed a long?). You can specify this when making a numpy array like: np.array([1,2,3],dtype=np.uint32)" + bcolors.ENDC)\n            return [-1, -1, -1]\n    else:\n        print(bcolors.FAIL + "dm (col steps) was not passed in (gave default value [1,0,1]) " + bcolors.ENDC)\n        dm = np.array([1, 0, 1], dtype=np.uint32)\n    # dw loading and exception handling\n    if \'dw\'  in parameter.keys():\n        try:\n            dw = np.array(parameter[\'dw\'], dtype=DTYPE_FLOAT)\n        except TypeError:\n            print(bcolors.FAIL + "FAILURE: The type of dw (step weights) is wrong - please pass in a 1-d numpy array that holds floats" + bcolors.ENDC)\n            return [-1, -1, -1]\n        except ValueError:\n            print(bcolors.FAIL + "FAILURE:The type of the elements in dw (step weights) is wrong - please have each element be a float (perhaps you passed ints or a long?). You can specify this when making a numpy array like: np.array([1,2,3],dtype=np.float64)" + bcolors.ENDC)\n            return [-1, -1, -1]\n    else:\n        dw = np.array([1, 1, 1], dtype=DTYPE_FLOAT)\n        print(bcolors.FAIL + "dw (step weights) was not passed in (gave default value [1,1,1]) " + bcolors.ENDC)\n\n    \n    \'\'\'\n    Section where types are given to the variables we\'re going to use \n    \'\'\'\n    # create matrices to store our results (D and E)\n    cdef DTYPE_INT32_t numRows = C.shape[0] # only works with np arrays, use np.shape(x) will work on lists? want to force to use np though?\n    cdef DTYPE_INT32_t numCols = C.shape[1]\n    cdef DTYPE_INT32_t numDifSteps = np.size(dw)\n\n    cdef unsigned int maxRowStep = max(dn)\n    cdef unsigned int maxColStep = max(dm)\n\n    cdef np.ndarray[np.uint32_t, ndim=2] steps = np.zeros((numRows,numCols), dtype=np.uint32)\n    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] accumCost = np.ones((maxRowStep + numRows, maxColStep + numCols), dtype=DTYPE_FLOAT) * MAX_FLOAT\n\n    cdef DTYPE_FLOAT_t bestCost\n    cdef DTYPE_INT32_t bestCostIndex\n    cdef DTYPE_FLOAT_t costForStep\n    cdef unsigned int row, col\n    cdef unsigned int stepIndex\n\n    \'\'\'\n    The start of the actual algorithm, now that all our variables are set up\n    \'\'\'\n    # initializing the cost matrix - depends on whether its subsequence DTW\n    # essentially allow us to hop on the bottom anywhere (so could start partway through one of the signals)\n    if parameter[\'SubSequence\']:\n        for col in range(numCols):\n            accumCost[maxRowStep, col + maxColStep] = C[0, col]\n    else:\n        accumCost[maxRowStep, maxColStep] = C[0,0]\n\n    # filling the accumulated cost matrix\n    for row in range(maxRowStep, numRows + maxRowStep, 1):\n        for col in range(maxColStep, numCols + maxColStep, 1):\n            bestCost = accumCost[<unsigned int>row, <unsigned int>col] # initialize with what\'s there - so if is an entry point, then can start low\n            bestCostIndex = 0\n            # go through each step, find the best one\n            for stepIndex in range(numDifSteps):\n                #costForStep = accumCost[<unsigned int>(row - dn[<unsigned int>(stepIndex)]), <unsigned int>(col - dm[<unsigned int>(stepIndex)])] + dw[<unsigned int>(stepIndex)] * C[<unsigned int>(row - maxRowStep), <unsigned int>(col - maxColStep)]\n                costForStep = accumCost[<unsigned int>((row - dn[(stepIndex)])), <unsigned int>((col - dm[(stepIndex)]))] + dw[stepIndex] * C[<unsigned int>(row - maxRowStep), <unsigned int>(col - maxColStep)]\n                if costForStep < bestCost:\n                    bestCost = costForStep\n                    bestCostIndex = stepIndex\n            # save the best cost and best cost index\n            accumCost[row, col] = bestCost\n            steps[<unsigned int>(row - maxRowStep), <unsigned int>(col - maxColStep)] = bestCostIndex\n\n    # return the accumulated cost along with the matrix of steps taken to achieve that cost\n    return [accumCost[maxRowStep:, maxColStep:], steps]\n\n@cython.boundscheck(False) # turn off bounds-checking for entire function\ndef DTW_GetPath(np.ndarray[DTYPE_FLOAT_t, ndim=2] accumCost, np.ndarray[np.uint32_t, ndim=2] stepsForCost, parameter):\n    \'\'\'\n\n    Parameter should have: \'dn\', \'dm\', \'dw\', \'SubSequence\'\n    \'\'\'\n\n    cdef np.ndarray[unsigned int, ndim=1] dn\n    cdef np.ndarray[unsigned int, ndim=1] dm\n    cdef np.uint8_t subseq\n    # make sure dn, dm, and dw are setup\n    if (\'dn\'  in parameter.keys()):\n        dn = parameter[\'dn\']\n    else:\n        dn = np.array([1, 1, 0], dtype=DTYPE_INT32)\n    if \'dm\'  in parameter.keys():\n        dm = parameter[\'dm\']\n    else:\n        dm = np.array([1, 0, 1], dtype=DTYPE_INT32)\n    if \'SubSequence\' in parameter.keys():\n        subseq = parameter[\'SubSequence\']\n    else:\n        subseq = 0\n\n    cdef np.uint32_t numRows\n    cdef np.uint32_t numCols\n    cdef np.uint32_t curRow\n    cdef np.uint32_t curCol\n    cdef np.uint32_t endCol\n    cdef DTYPE_FLOAT_t endCost\n\n    numRows = accumCost.shape[0]\n    numCols = accumCost.shape[1]\n\n    # either start at the far corner (non sub-sequence)\n    # or start at the lowest cost entry in the last row (sub-sequence)\n    # where all of the signal along the row has been used, but only a \n    # sub-sequence of the signal along the columns has to be used\n    curRow = numRows - 1\n    if subseq:\n        curCol = np.argmin(accumCost[numRows - 1, :])\n    else:\n        curCol = numCols - 1\n\n    endCol = curCol\n    endCost = accumCost[curRow, curCol]\n\n    cdef np.uint32_t curRowStep\n    cdef np.uint32_t curColStep\n    cdef np.uint32_t curStepIndex\n\n\n    cdef np.ndarray[np.uint32_t, ndim=2] path = np.zeros((2, numRows + numCols), dtype=np.uint32) # make as large as could need, then chop at the end\n    path[0, 0] = curRow\n    path[1, 0] = curCol\n\n    cdef np.uint32_t stepsInPath = 1 # starts at one, we add in one before looping\n    cdef np.uint32_t stepIndex = 0\n    cdef np.int8_t done = (subseq and curRow == 0) or (curRow == 0 and curCol == 0)\n    while not done:\n        if accumCost[curRow, curCol] == MAX_FLOAT:\n            print(\'A path is not possible\')\n            break\n\n        # you\'re done if you\'ve made it to the bottom left (non sub-sequence)\n        # or just the bottom (sub-sequence)\n        # find the step size\n        curStepIndex = stepsForCost[curRow, curCol]\n        curRowStep = dn[curStepIndex]\n        curColStep = dm[curStepIndex]\n        # backtrack by 1 step\n        curRow = curRow - curRowStep\n        curCol = curCol - curColStep\n        # add your new location onto the path\n        path[0, stepsInPath] = curRow\n        path[1, stepsInPath] = curCol\n        stepsInPath = stepsInPath + 1\n        # check to see if you\'re done\n        done = (subseq and curRow == 0) or (curRow == 0 and curCol == 0)\n\n    # reverse the path (a matrix with two rows) and return it\n    return [np.fliplr(path[:, 0:stepsInPath]), endCol, endCost]\n\nclass bcolors:\n    HEADER = \'\\033[95m\'\n    OKBLUE = \'\\033[94m\'\n    OKGREEN = \'\\033[92m\'\n    WARNING = \'\\033[93m\'\n    FAIL = \'\\033[91m\'\n    ENDC = \'\\033[0m\'\n    BOLD = \'\\033[1m\'\n    UNDERLINE = \'\\033[4m\'')


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
    D = np.zeros(C.shape)
    B = np.zeros((C.shape[0],C.shape[1],2))

    ### START CODE BLOCK ###
    inf = float("inf")
    D[0,:]=C[0,:]
    cur_segment = 0
    for i in range(1, D.shape[0]):
        for j in range(D.shape[1]):
            steps.append([1,int(segment_lengths[i]/2)])
            weights.append(1)
            opt=[inf for i in range(len(steps))]
            for index, s in enumerate(steps):
                if i-s[0] >= 0 and j-s[1]>=0 and D[i-s[0],j-s[1]]!=inf:
                    previousCell = D[i-s[0],j-s[1]]
                    opt[index] = previousCell +C[i,j]*weights[index]
            optimal = min(opt)
            opt_index = np.argmin(np.array(opt))
            D[i][j]=optimal
            B[i][j][0]=steps[opt_index][0]
            B[i][j][1]=steps[opt_index][1]
            if opt_index == 1:
              cur_segment+=1
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
        if r==0:
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




